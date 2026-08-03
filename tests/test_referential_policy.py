"""Referential policy, and the one deletion model both layers agree on.

The application deletes through `OwnershipPath` and verifies independently.
The database enforces foreign keys. Those are two mechanisms that could easily
become two *models*: a cascade would remove rows the application's verification
never saw and report success either way, and nothing would notice until someone
asked what had actually been deleted.

So every relationship is RESTRICT. The database does not delete anything; it
refuses to let a parent go while a dependent survives. That turns the
application's ordering from a convention into something enforced — and this file
checks the two against each other rather than trusting that they match.

The deletion order itself is derived from the relationship graph. It used to be
a heuristic (indirectly-owned tables first), which was correct for the single
indirect table that existed and says nothing about a dependency between two
directly-owned tables, which `event_reconciliation` referencing its events is.
"""
from __future__ import annotations

import os

import pytest

from src.db import schema
from src.db.engine import Database
from src.workspace.erasure import delete_workspace, export_workspace, verify_deleted
from src.workspace.retention import WORKSPACE_RECORDS, OwnerScope
from src.workspace.store import WorkspaceStore

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

OWNER = "alice"


@pytest.fixture
def postgres_store():
    if not POSTGRES_URL:
        pytest.skip("set QUANTIFY_TEST_POSTGRES_URL to run against PostgreSQL")
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    return WorkspaceStore(POSTGRES_URL)


@pytest.fixture
def sqlite_store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def populate(store, owner=OWNER):
    """A workspace with a parent and a dependent in every declared relationship."""
    from src.mission.rsu_reconcile import (
        EventReconciliation, ObservedEvent, PlannedEvent, ReconciliationStatus)

    store.record_planned_event(
        owner=owner, worksheet_id="ws-1",
        event=PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                           expected_date="2026-06-15", employer_asset="ACME",
                           expected_gross_shares="152.26"),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1")
    store.record_observed_event(
        owner=owner, worksheet_id="ws-1",
        event=ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                            effective_date="2026-06-15", grant_ref="grant/g1",
                            employer_asset="ACME", gross_shares="152.26"),
        created_at="2026-01-01T00:00:00Z")
    store.record_reconciliation(
        owner=owner, worksheet_id="ws-1",
        reconciliation=EventReconciliation(
            reconciliation_id="rc-1", status=ReconciliationStatus.MATCHED,
            planned_ref="pe-1", observed_ref="oe-1",
            derived_at="2026-06-17T00:00:00Z"))
    return store


class TestTheOrderIsDerivedNotAssumed:
    def test_every_table_appears_exactly_once(self):
        order = schema.deletion_order()
        assert sorted(order) == sorted(schema.metadata.tables)

    def test_each_dependent_precedes_its_parent(self):
        order = schema.deletion_order()
        for one in schema.RELATIONSHIPS:
            assert order.index(one.table) < order.index(one.parent), (
                f"{one.table} must be deleted before {one.parent}, or the "
                "parent delete is refused and the deletion fails halfway")

    def test_a_cycle_is_refused_rather_than_ordered_arbitrarily(self,
                                                               monkeypatch):
        """An arbitrary order would fail at a different table each run."""
        cycle = (
            schema.Relationship(table="plan", columns=("plan_id",),
                                parent="plan_run", parent_columns=("run_id",),
                                policy=schema.DeletePolicy.RESTRICT,
                                rationale="synthetic"),
        ) + schema.RELATIONSHIPS
        monkeypatch.setattr(schema, "RELATIONSHIPS", cycle)
        with pytest.raises(RuntimeError, match="cycle"):
            schema.deletion_order()


class TestTheApplicationPathAndTheDatabaseAgree:
    """One deletion model, checked from both ends."""

    @pytest.mark.parametrize("engine", ["sqlite", "postgres"])
    def test_the_application_path_empties_the_workspace(
            self, engine, sqlite_store, request):
        store = (sqlite_store if engine == "sqlite"
                 else request.getfixturevalue("postgres_store"))
        populate(store)
        receipt = delete_workspace(store, OWNER,
                                   requested_at="2026-07-01T00:00:00Z")
        assert receipt.status == "COMPLETE"
        assert verify_deleted(store, OWNER) == {}

    @pytest.mark.parametrize("engine", ["sqlite", "postgres"])
    def test_the_database_refuses_what_the_application_orders_around(
            self, engine, sqlite_store, request):
        """Delete a parent first and the database must refuse.

        This is what makes the application's ordering a guarantee rather than a
        habit: skip a step and the failure is loud, not a silently orphaned row.
        """
        store = (sqlite_store if engine == "sqlite"
                 else request.getfixturevalue("postgres_store"))
        populate(store)
        with pytest.raises(Exception) as caught:
            with store._conn() as conn:
                conn.execute("DELETE FROM planned_event WHERE owner = ?",
                             (OWNER,))
        assert "foreign key" in str(caught.value).lower() or \
               "violates" in str(caught.value).lower()

    @pytest.mark.parametrize("engine", ["sqlite", "postgres"])
    def test_nothing_survives_that_the_ownership_graph_did_not_reach(
            self, engine, sqlite_store, request):
        """The classification drives deletion; this checks every classified
        table independently rather than trusting the receipt."""
        store = (sqlite_store if engine == "sqlite"
                 else request.getfixturevalue("postgres_store"))
        populate(store)
        delete_workspace(store, OWNER, requested_at="2026-07-01T00:00:00Z")
        with store._conn() as conn:
            for record in WORKSPACE_RECORDS.values():
                if record.owner_scope is OwnerScope.DIRECT:
                    rows = conn.execute(
                        f"SELECT COUNT(*) AS n FROM {record.table} "
                        f"WHERE {record.owner_column} = ?", (OWNER,)).fetchone()
                    assert rows["n"] == 0, f"{record.table} kept rows"

    @pytest.mark.parametrize("engine", ["sqlite", "postgres"])
    def test_another_owner_is_untouched(self, engine, sqlite_store, request):
        store = (sqlite_store if engine == "sqlite"
                 else request.getfixturevalue("postgres_store"))
        populate(store, owner=OWNER)
        populate(store, owner="bob")
        delete_workspace(store, OWNER, requested_at="2026-07-01T00:00:00Z")
        assert export_workspace(store, "bob")["counts"]["planned_event"] == 1

    def test_no_relationship_cascades(self):
        """A cascade would delete rows the application's verification never
        saw, and both layers would report success."""
        assert all(one.policy is schema.DeletePolicy.RESTRICT
                   for one in schema.RELATIONSHIPS)


class TestTelemetryReferencesAreNotConstrained:
    def test_trace_id_has_no_foreign_key(self):
        """Traces expire on their own schedule. A constraint would either block
        that expiry or take research records with it, and a trace that has aged
        out must leave a readable intent behind."""
        constrained = {(one.table, column)
                       for one in schema.RELATIONSHIPS for column in one.columns}
        assert ("worksheet_intent", "trace_id") not in constrained
        assert ("worksheet_proposal", "trace_id") not in constrained


class TestStatusVocabulary:
    """Vocabulary only. Lifecycle stays in application code."""

    def test_each_declared_value_is_accepted(self, sqlite_store):
        # The observation has to exist: the reconciliation's reference to it is
        # a real constraint, not a label. Writing "oe-x" here was refused, which
        # is the foreign key doing its job.
        populate(sqlite_store)
        without_observation = ("PENDING", "UNOBSERVED_OVERDUE",
                               "MISSING_CONFIRMED")
        for index, status in enumerate(
                schema.STATUS_VOCABULARY["event_reconciliation"]):
            with sqlite_store._conn() as conn:
                conn.execute(
                    "INSERT INTO event_reconciliation (owner, worksheet_id, "
                    "reconciliation_id, status, payload, "
                    "matching_policy_version, content_hash, derived_at, "
                    "observed_event_id) VALUES (?,?,?,?,?,?,?,?,?)",
                    (OWNER, "ws-1", f"rc-vocab-{index}", status, "{}", "m@1",
                     "h", "2026-01-01T00:00:00Z",
                     None if status in without_observation else "oe-1"))

    def test_an_undeclared_value_is_refused(self, sqlite_store):
        with pytest.raises(Exception):
            with sqlite_store._conn() as conn:
                conn.execute(
                    "INSERT INTO worksheet_proposal (proposal_id, owner, "
                    "worksheet_id, source_revision, status, payload, "
                    "created_at) VALUES (?,?,?,?,?,?,?)",
                    ("wp-1", OWNER, "ws-1", 1, "MAYBE", "{}",
                     "2026-01-01T00:00:00Z"))

    def test_the_vocabulary_tracks_the_enum(self):
        """Retyped in the schema, the list would fall behind the enum silently
        and start rejecting a status the application had begun to write."""
        from src.workspace.apply import ProposalStatus
        assert set(schema.STATUS_VOCABULARY["worksheet_proposal"]) == {
            member.value for member in ProposalStatus}

    def test_lifecycle_is_not_encoded_in_the_database(self):
        """PROPOSED -> ACCEPTED needs facts the row does not carry."""
        constraints = {
            c.name for c in
            schema.metadata.tables["worksheet_proposal"].constraints
            if c.name}
        assert not any("transition" in (name or "") for name in constraints)


class TestLocalRowConsistency:
    def test_a_result_revision_requires_acceptance(self, sqlite_store):
        """A rejected proposal carrying a revision would claim an edit was
        applied that the worksheet history does not show."""
        with pytest.raises(Exception):
            with sqlite_store._conn() as conn:
                conn.execute(
                    "INSERT INTO worksheet_proposal (proposal_id, owner, "
                    "worksheet_id, source_revision, status, payload, "
                    "created_at, result_revision) VALUES (?,?,?,?,?,?,?,?)",
                    ("wp-1", OWNER, "ws-1", 1, "REJECTED", "{}",
                     "2026-01-01T00:00:00Z", 2))

    def test_acceptance_with_a_revision_is_allowed(self, sqlite_store):
        with sqlite_store._conn() as conn:
            conn.execute(
                "INSERT INTO worksheet_proposal (proposal_id, owner, "
                "worksheet_id, source_revision, status, payload, created_at, "
                "result_revision) VALUES (?,?,?,?,?,?,?,?)",
                ("wp-1", OWNER, "ws-1", 1, "ACCEPTED", "{}",
                 "2026-01-01T00:00:00Z", 2))

    def test_a_conclusion_without_an_observation_is_refused(self, sqlite_store):
        """MATCHED with no observed event is a conclusion drawn from nothing —
        `unknown is not false`, enforced at rest."""
        with pytest.raises(Exception):
            with sqlite_store._conn() as conn:
                conn.execute(
                    "INSERT INTO event_reconciliation (owner, worksheet_id, "
                    "reconciliation_id, status, payload, "
                    "matching_policy_version, content_hash, derived_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (OWNER, "ws-1", "rc-1", "MATCHED", "{}", "m@1", "h",
                     "2026-01-01T00:00:00Z"))

    @pytest.mark.parametrize("status", ["PENDING", "UNOBSERVED_OVERDUE",
                                        "MISSING_CONFIRMED"])
    def test_the_three_absence_states_may_have_no_observation(
            self, sqlite_store, status):
        """Each means something specific about the absence, and none of them is
        the same as the others."""
        with sqlite_store._conn() as conn:
            conn.execute(
                "INSERT INTO event_reconciliation (owner, worksheet_id, "
                "reconciliation_id, status, payload, matching_policy_version, "
                "content_hash, derived_at) VALUES (?,?,?,?,?,?,?,?)",
                (OWNER, "ws-1", f"rc-{status}", status, "{}", "m@1", "h",
                 "2026-01-01T00:00:00Z"))
