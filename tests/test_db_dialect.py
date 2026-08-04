"""The dialect layer, checked against the SQL the store actually issues.

Two independent sources of truth are kept apart on purpose:

    `src/db/schema.py`        the model Alembic and both dialects render from
    a live SQLite database    what the shipped DDL actually produced

Comparing the model against itself would pass no matter what the model said.
Comparing it against a database built by the original `_SCHEMA` string proves
the rewrite preserved the schema rather than merely restating it.

The translation tests read the store's source for its upsert statements rather
than repeating them here. Parametrising from a copied list would let a new
untranslatable statement pass by never appearing in the list — the same hole
the comparison-profile and diagnostic-destination guards had to close.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from src.db import schema
from src.db.engine import (
    Database,
    Dialect,
    UnsupportedTarget,
    UntranslatableStatement,
    dialect_of,
    resolve_target,
    to_postgres,
)
from src.workspace.store import _SCHEMA as SHIPPED_DDL
from src.workspace.store import WorkspaceStore

STORE_SOURCE = Path("src/workspace/store.py").read_text()


def upsert_statements():
    """Every `INSERT OR REPLACE` the store issues, read from the store.

    Sourced from the code rather than a list so a newly added statement is
    covered the moment it is written.
    """
    return re.findall(r'"""(INSERT OR REPLACE.*?)"""', STORE_SOURCE, re.DOTALL)


class TestTheModelMatchesTheShippedSchema:
    """The rewrite must preserve the schema, not just describe one."""

    @pytest.fixture(scope="class")
    def shipped(self, tmp_path_factory):
        """A database built from the original DDL, not from the model.

        Building this with `WorkspaceStore` would create the tables *from
        `src/db/schema.py`* and then compare them against it — a test that
        passes whatever the model says. Dropping a column from the model was
        caught only once this fixture stopped going through the store.

        When `_SCHEMA` is finally deleted, the independent oracle becomes the
        Alembic baseline migration, and `tests/test_migration_parity.py` takes
        this role over.
        """
        path = tmp_path_factory.mktemp("shipped") / "w.db"
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SHIPPED_DDL)
        return conn

    #: Tables introduced after `_SCHEMA` was shipped, with the reason. The
    #: oracle is the *shipped* schema, so a legitimate new table appears here
    #: as a declaration rather than by widening the comparison — an accidental
    #: table still fails.
    DELIBERATELY_ADDED_TABLES = {
        "market_data_access_event":
            "the factual record that one execution received one realized "
            "frame. A stored run previously cited only what its producer "
            "declared it had used, and the producer is the component a defect "
            "would corrupt.",
    }

    def test_every_table_is_modelled(self, shipped):
        live = {row["name"] for row in shipped.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")}
        modelled = set(schema.metadata.tables)
        added = (modelled - live) & set(self.DELIBERATELY_ADDED_TABLES)
        assert live == modelled - added, (
            "the model and the database disagree about which tables exist")

    def test_each_added_table_records_why(self):
        for table, reason in self.DELIBERATELY_ADDED_TABLES.items():
            assert table in schema.metadata.tables, f"{table} no longer exists"
            assert len(reason.strip()) > 40, table

    #: Columns added after shipping, with the reason. Enumerated so an
    #: accidental addition still fails the comparison.
    DELIBERATELY_ADDED = {
        # `plan_run` was reachable only through its plan. Every ownership
        # question was a join, and deletion was one forgotten cascade away
        # from keeping every run while reporting success.
        ("plan_run", "owner"),
        # Which delivery produced these figures. Nullable: runs recorded before
        # deliveries were captured cite none, and back-filling one from today's
        # configuration would manufacture the evidence the column exists to
        # provide.
        ("plan_run", "access_event_id"),
    }

    def test_every_column_is_modelled(self, shipped):
        for table in sorted(schema.metadata.tables):
            if table in self.DELIBERATELY_ADDED_TABLES:
                # No shipped counterpart to compare against; the table itself
                # is declared above, and its shape is the model's own.
                continue
            live = {row["name"] for row in
                    shipped.execute(f"PRAGMA table_info({table})")}
            modelled = {column.name
                        for column in schema.metadata.tables[table].columns}
            added = {name for name in modelled - live
                     if (table, name) in self.DELIBERATELY_ADDED}
            assert live == modelled - added, f"{table} columns disagree"

    #: Tables whose primary key the model deliberately widened after shipping,
    #: with the column added. Each was keyed without `owner`, so two tenants
    #: could not hold the same id — and because these tables are written with
    #: `INSERT OR REPLACE`, one tenant's write silently overwrote the other's
    #: row rather than merely being refused. The same defect was fixed for
    #: `worksheet` when it was found there; it was never propagated.
    #:
    #: Enumerated rather than skipped, so an *accidental* primary-key change
    #: still fails this test.
    DELIBERATELY_WIDENED = {
        "worksheet_proposal": "owner",
        "proposal": "owner",
        "observation": "owner",
        "worksheet_intent": "owner",
        "confirmation_event": "owner",
        "plan": "owner",
        # `plan_run` gained the column outright rather than widening an
        # existing one, so it is compared separately below.
    }

    def test_primary_keys_match(self, shipped):
        """The conflict target of every upsert depends on this being right."""
        for table in sorted(schema.metadata.tables):
            if table in self.DELIBERATELY_ADDED_TABLES:
                # Asserted directly, for the same reason as `plan_run` below:
                # there is no shipped key to compare against.
                assert schema.primary_key_columns(table)[0] == "owner", (
                    f"{table} is not keyed by owner first")
                continue
            if table == "plan_run":
                # Had no owner column at all before the ownership migration,
                # so there is no shipped key to compare against. Its identity
                # is asserted directly instead.
                assert schema.primary_key_columns(table) == ("owner", "run_id")
                continue
            columns = shipped.execute(f"PRAGMA table_info({table})").fetchall()
            live = tuple(c["name"] for c in
                         sorted((c for c in columns if c["pk"]),
                                key=lambda c: c["pk"]))
            modelled = schema.primary_key_columns(table)
            added = self.DELIBERATELY_WIDENED.get(table)
            if added is not None:
                assert added in modelled, (
                    f"{table} was widened to include {added!r} and no longer is")
                modelled = tuple(c for c in modelled if c != added)
            assert live == modelled, (
                f"{table} primary key disagrees; an upsert would target the "
                "wrong columns and silently insert a duplicate")

    def test_every_owner_scoped_table_keys_by_owner(self):
        """The standing rule, checked against the model rather than remembered.

        `worksheet` got `owner` in its key when a write refusal was found to be
        answering a question about another tenant. Five more tables had the same
        shape and kept it, because nothing enumerated them — so the rule was
        satisfied where it had been applied and nowhere else.
        """
        unscoped = []
        for name, table in schema.metadata.tables.items():
            if "owner" not in table.columns:
                continue                      # indirectly owned; see retention
            if "owner" not in schema.primary_key_columns(name):
                unscoped.append(name)
        assert unscoped == [], (
            "owner-scoped tables whose identity omits the owner: "
            f"{unscoped}. Two tenants cannot then hold the same id, and an "
            "`INSERT OR REPLACE` lets one overwrite the other")

    def test_foreign_keys_are_not_lost(self, shipped):
        """The shipped schema's constraints must survive the rewrite.

        This test exists because they did not. `plan_run` carried
        `FOREIGN KEY (plan_id) REFERENCES plan (plan_id)` in the original DDL,
        the metadata rewrite dropped it, and every parity check passed — they
        compared tables, columns, keys and nullability, and a constraint is none
        of those. A schema comparison that only looks at shape will keep missing
        the parts that carry the guarantees.
        """
        for table in sorted(schema.metadata.tables):
            live = {(row["from"], row["table"], row["to"]) for row in
                    shipped.execute(f"PRAGMA foreign_key_list({table})")}
            modelled = {
                (column, one.parent, parent)
                for one in schema.RELATIONSHIPS if one.table == table
                for column, parent in zip(one.columns, one.parent_columns)}
            assert live <= modelled, (
                f"{table} had foreign keys the model does not declare: "
                f"{live - modelled}")

    def test_every_relationship_reaches_the_metadata(self):
        """A declared relationship that never became a constraint is a policy
        nothing enforces."""
        for one in schema.RELATIONSHIPS:
            constraints = schema.metadata.tables[one.table].foreign_key_constraints
            targets = {tuple(sorted(fk.column.name for fk in c.elements))
                       for c in constraints}
            assert tuple(sorted(one.parent_columns)) in targets, (
                f"{one.table} declares a relationship to {one.parent} that no "
                "constraint implements")

    def test_delete_policies_are_stated_not_defaulted(self):
        """A blanket CASCADE would make the database a second deletion model,
        and its version would win silently."""
        for one in schema.RELATIONSHIPS:
            assert one.policy in schema.DeletePolicy
            assert one.rationale.strip(), (
                f"{one.table} -> {one.parent} has a delete policy and no "
                "recorded reason for it")

    def test_nullability_matches(self, shipped):
        """`trial_effect` nullable is a decision, not an accident."""
        for table in sorted(schema.metadata.tables):
            for row in shipped.execute(f"PRAGMA table_info({table})"):
                modelled = schema.metadata.tables[table].columns[row["name"]]
                # A primary key column is implicitly NOT NULL in both.
                if modelled.primary_key:
                    continue
                assert bool(row["notnull"]) is not modelled.nullable, (
                    f"{table}.{row['name']} nullability disagrees")


class TestUpsertTranslation:
    def test_every_store_upsert_translates(self):
        statements = upsert_statements()
        assert statements, "found no upsert statements — the pattern is wrong"
        for statement in statements:
            translated = to_postgres(statement)
            assert "OR REPLACE" not in translated
            assert "ON CONFLICT" in translated
            assert "?" not in translated

    def test_conflict_target_is_the_primary_key(self):
        translated = to_postgres(
            "INSERT OR REPLACE INTO worksheet "
            "(worksheet_id, revision, owner, payload, canonical_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)")
        assert "ON CONFLICT (owner, worksheet_id, revision)" in translated

    def test_key_columns_are_not_assigned(self):
        """Assigning a key column in the update is a no-op at best.

        `owner` is a key column on `plan` since the ownership migration, so it
        belongs on the excluded side now — which is what the conflict target
        being read from the model rather than a list gets right for free.
        """
        translated = to_postgres(
            "INSERT OR REPLACE INTO plan (plan_id, owner, title) "
            "VALUES (?, ?, ?)")
        assignments = translated.split("DO UPDATE SET")[1]
        assert "plan_id = EXCLUDED" not in assignments
        assert "owner = EXCLUDED" not in assignments
        assert "title = EXCLUDED.title" in assignments
        assert "ON CONFLICT (plan_id, owner)" in translated

    def test_an_untranslatable_statement_is_refused(self):
        """Passing it through would surface a driver syntax error instead."""
        with pytest.raises(UntranslatableStatement):
            to_postgres("INSERT OR REPLACE INTO plan SELECT * FROM plan__old")


class TestPlaceholders:
    def test_placeholders_become_percent_s(self):
        assert to_postgres("SELECT * FROM plan WHERE owner = ?") == (
            "SELECT * FROM plan WHERE owner = %s")

    def test_a_question_mark_inside_a_literal_is_left_alone(self):
        """A literal `?` is data. Rewriting it would corrupt the value."""
        assert to_postgres("SELECT * FROM plan WHERE title = '?' AND owner = ?") == (
            "SELECT * FROM plan WHERE title = '?' AND owner = %s")


class TestTargetResolution:
    def test_a_path_stays_sqlite(self, tmp_path):
        assert resolve_target(tmp_path / "w.db").startswith("sqlite:///")

    def test_a_url_is_passed_through(self):
        url = "postgresql://user:pw@host:5432/db"
        assert resolve_target(url) == url

    def test_the_environment_decides_when_nothing_is_given(self, monkeypatch):
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", "postgresql://h/db")
        assert resolve_target() == "postgresql://h/db"

    def test_dialects_are_recognised(self):
        assert dialect_of("sqlite:///data/w.db") is Dialect.SQLITE
        assert dialect_of("postgresql://h/db") is Dialect.POSTGRESQL

    def test_an_unknown_engine_is_refused(self):
        """Naming the engine beats a driver import error three frames down."""
        with pytest.raises(UnsupportedTarget):
            dialect_of("mysql://host/db")


class TestSqliteRemainsIntact:
    def test_a_store_opens_and_round_trips(self, tmp_path):
        store = WorkspaceStore(tmp_path / "w.db")
        assert store.db.dialect is Dialect.SQLITE
        store.record_confirmation_event(
            event_id="ev-1", owner="alice", occurred_at="2026-01-01T00:00:00Z",
            kind="EDIT", path="/w/1", field="rate", provenance="USER",
            original_value="1", final_value="2", reason="", compiler_version="c@1",
            defaults_ref="d@1")
        assert len(store.confirmation_events("alice")) == 1
        assert store.confirmation_events("bob") == []

    def test_reopening_an_existing_database_is_safe(self, tmp_path):
        """`create_all` must not disturb a database that already has tables."""
        path = tmp_path / "w.db"
        WorkspaceStore(path).record_confirmation_event(
            event_id="ev-1", owner="alice", occurred_at="2026-01-01T00:00:00Z",
            kind="EDIT", path=None, field=None, provenance=None,
            original_value=None, final_value=None, reason=None,
            compiler_version=None, defaults_ref=None)
        assert len(WorkspaceStore(path).confirmation_events("alice")) == 1
