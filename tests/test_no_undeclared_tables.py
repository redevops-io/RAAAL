"""Every table the application creates must be one the model declares.

`pilot_plans` was created by a `CREATE TABLE IF NOT EXISTS` in the store and by
no migration. It therefore appeared in production the first time somebody saved
a runtime plan — partway through the life of a deployment, long after the
deploy that shipped the code that creates it.

Nothing noticed for days, because a running process does not re-run its
preflight. The schema-parity check refuses to serve against a database it
cannot account for, and at startup it was satisfied: the table did not exist
yet. The refusal arrived at the next restart, on an unrelated deploy, as
`SCHEMA_MISMATCH ('remove_table', pilot_plans)` — pointing at the deploy that
surfaced it rather than the feature that caused it.

A table created on first use is a landmine whose fuse is however long it takes
somebody to restart.

The check reads string *constants* through the AST rather than grepping the
files. Grepped, this test's own explanation would count as a violation — the
sentence above contains the syntax it forbids.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"

#: `CREATE TABLE [IF NOT EXISTS] <name>`, capturing the name.
STATEMENT = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE)


def docstrings(tree: ast.AST) -> set[int]:
    """The id() of every string constant that is a docstring.

    Excluded because a docstring explaining this rule is prose, not code —
    which is the mistake the module docstring above describes.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                found.add(id(body[0].value))
    return found


def created_tables():
    """(file, table) for every table the application source creates itself."""
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a file that will not parse
            continue
        skip = docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in skip:
                for name in STATEMENT.findall(node.value):
                    found.append((path.relative_to(ROOT), name))
    return found


#: Modules that keep their own database file rather than using the
#: application's. Their tables are correctly absent from the model, and the
#: parity preflight never looks at them.
#:
#: Named individually. A blanket exemption for "anything the model does not
#: know" would exempt the very fault this file exists for — the point is that
#: a table in the application database must be declared, and these are not in
#: it. Each connects with `sqlite3.connect(self.path)`, which is what makes
#: them a different database rather than a different opinion.
SEPARATE_DATABASES = {
    "src/ledger.py",
    "src/telemetry/trace_store.py",
}


def declared():
    from src.db.schema import metadata

    return set(metadata.tables)


@pytest.mark.skipif(not SOURCE.exists(), reason="no source tree here")
class TestNothingCreatesATableTheModelDoesNotKnow:
    def test_the_scan_finds_something(self):
        """If this stops finding statements the check has gone blind — either
        the source moved or the pattern stopped matching."""
        assert created_tables(), (
            "found no CREATE statements in the source; this check is stale")

    def test_every_created_table_is_declared(self):
        known = declared()
        undeclared = sorted({
            f"{name} (in {path})" for path, name in created_tables()
            if name not in known and str(path) not in SEPARATE_DATABASES
            # `IF` is what the pattern captures when a statement is assembled
            # from a format string rather than written whole. Not a table.
            and name.upper() not in {"IF", "NOT", "EXISTS"}})
        assert not undeclared, (
            "these tables are created by the application and declared nowhere "
            f"in src/db/schema.py: {undeclared}. They appear in a database the "
            "first time the feature is used, so the schema-parity preflight "
            "refuses to start — at the next restart, which may be days later "
            "and on an unrelated deploy")

    def test_the_runtime_plan_table_is_among_them(self):
        """The specific one, named. This is the table that took the site down,
        and a regression here should say so rather than reporting a list."""
        assert "pilot_plans" in declared(), (
            "pilot_plans is not declared in the model. The runtime store "
            "creates it on first use; undeclared, it fails the preflight at "
            "the next restart")


@pytest.mark.skipif(not SOURCE.exists(), reason="no source tree here")
class TestEveryDeclaredTableHasAMigration:
    """The other direction: declared and never created is equally broken.

    A table in the model that no migration builds fails parity on a fresh
    database — the mirror image, and the one a new environment meets first.
    """

    def test_the_runtime_plan_table_is_migrated(self):
        migrations = ROOT / "migrations" / "versions"
        if not migrations.exists():
            pytest.skip("no migrations here")
        created = " ".join(
            path.read_text(encoding="utf-8") for path in migrations.glob("*.py"))
        assert "pilot_plans" in created, (
            "pilot_plans is declared in the model and no migration creates "
            "it, so a database built from migrations alone would not have it")
