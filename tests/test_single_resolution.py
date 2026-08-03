"""Questions several components depend on are answered once.

`WorkspaceStore()` substituted a default before the resolver could read the
configured database URL. The preflight validated PostgreSQL and the application
wrote to a local SQLite file — both correct about the question they asked, and
nothing asked whether they had asked the same question.

So the readers are enumerated. A new component resolving an operational
identity for itself fails here, rather than being discovered when it disagrees
with something in production.

This is a weaker check than the invariant deserves: the honest form is a
resolved deployment context that everything else consumes, with nothing below
it touching the environment. Until that exists, this at least makes each direct
reader a decision someone made rather than one nobody noticed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Operational identities that must be resolved once.
RESOLVED_IDENTITIES = {
    "QUANTIFY_DATABASE_URL": "src/db/engine.py::resolve_target",
    "PILOT_DATA_POLICY": "src/market_data/pilot_policy.py::configured_policy",
    "QUANTIFY_DEPLOYMENT_PROFILE": "src/deploy/preflight.py::configured_profile",
}

#: Modules permitted to read one directly, and why.
#:
#: `preflight.py` reads the database URL and hands it to `resolve_target`
#: rather than calling it with `None`. That is one read too many — it agrees
#: with the store because both end at the same function, not because the
#: environment is consulted once. It is listed rather than hidden.
DECLARED_READERS = {
    ("src/db/engine.py", "QUANTIFY_DATABASE_URL"):
        "the resolver itself",
    ("src/deploy/preflight.py", "QUANTIFY_DATABASE_URL"):
        "reads it to distinguish 'unset' from 'set to something unusable', "
        "which the resolver cannot express because it falls back. Should "
        "become a resolved deployment context.",
    ("src/deploy/preflight.py", "QUANTIFY_DEPLOYMENT_PROFILE"):
        "the resolver itself",
    ("src/market_data/pilot_policy.py", "PILOT_DATA_POLICY"):
        "the resolver itself",
    ("src/deploy/manifest.py", "QUANTIFY_DEPLOYMENT_PROFILE"):
        "reports the profile as an optional deployment fact; does not act on it",
    ("src/deploy/manifest.py", "PILOT_DATA_POLICY"):
        "reports the configured policy as an optional deployment fact so a "
        "build can say which one it was started under. It reports and does not "
        "decide — `configured_policy` is what any gate consults.",
}


def environment_readers():
    """Every `src/` module naming one of these variables, from the AST."""
    found = set()
    for path in sorted(Path("src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in RESOLVED_IDENTITIES:
                    found.add((str(path), node.value))
    return found


class TestEveryDirectReaderIsDeclared:
    def test_no_undeclared_reader(self):
        undeclared = environment_readers() - set(DECLARED_READERS)
        assert undeclared == set(), (
            f"these read an operational identity directly: {sorted(undeclared)}. "
            "Take the resolved object instead, or declare why this one cannot")

    def test_each_declaration_records_why(self):
        for key, reason in DECLARED_READERS.items():
            assert reason.strip(), key

    def test_the_scan_is_not_vacuous(self):
        """A scan finding nothing would pass the check above."""
        assert environment_readers()

    def test_the_store_is_not_among_them(self):
        """The defect: the store answering the question for itself."""
        readers = {module for module, _ in environment_readers()}
        assert "src/workspace/store.py" not in readers, (
            "the store resolves its own database target again; it must take "
            "what the resolver returns")

    def test_no_route_reads_an_operational_identity(self):
        """A request handler deciding which database or policy applies would be
        the same defect at the surface."""
        readers = {module for module, _ in environment_readers()}
        for route in ("src/workspace/routes.py", "src/web/routes.py"):
            assert route not in readers, route


class TestTheResolversAreNamed:
    def test_each_identity_names_where_it_is_resolved(self):
        for variable, location in RESOLVED_IDENTITIES.items():
            module, _, function = location.partition("::")
            assert Path(module).exists(), variable
            assert function, variable

    def test_each_named_resolver_exists(self):
        for variable, location in RESOLVED_IDENTITIES.items():
            module, _, function = location.partition("::")
            tree = ast.parse(Path(module).read_text())
            names = {node.name for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            assert function in names, f"{location} does not exist"
