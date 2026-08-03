"""Deleting one tenant's workspace, with three authorities that must agree.

    OwnershipPath        which rows belong to the tenant
    dependency graph     the order they can be removed in
    PostgreSQL RESTRICT  refusal when that order is wrong

Each is checked on its own. Together they could pass for the wrong reason: a
deletion that happened to enumerate the right rows would look identical to one
whose ownership joins were correct, until the day a join returned another
tenant's row instead. So Bob exists throughout with byte-for-byte identical
identifiers, and every assertion is made from a fresh session after commit.

The mutation this file is built to catch is not a missing table but a *wrong*
join — one that still returns rows, and returns the wrong tenant's.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import os

import pytest

from src.db.engine import Database
from src.mission.rsu_reconcile import (
    EventReconciliation,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
)
from src.workspace import retention
from src.workspace.erasure import (
    DeletionIncomplete,
    delete_workspace,
    export_workspace,
    verify_deleted,
)
from src.workspace.retention import (
    OwnerScope,
    OwnershipPath,
    WORKSPACE_RECORDS,
)
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

from tests import ownership_fixture

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; deletion under real foreign keys "
           "is a PostgreSQL-only guarantee")

A, B = "alice", "bob"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}

#: Identical for both tenants, on purpose.
PLAN = "p-shared"
RUN = "r-shared"
WORKSHEET = "ws-shared"
PROPOSAL = "wp-shared"
CHILD = "child-shared"


def session():
    return WorkspaceStore(POSTGRES_URL)


def observe(query, params=()):
    conn = Database(POSTGRES_URL).connect()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def seed(owner):
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    store = session()
    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name=PLAN, version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id=PLAN, owner=owner, scenario=scenario,
                    stated_text="seed", saved_at="2026-01-01T00:00:00Z")
    store.record_run(run_id=RUN, plan_id=PLAN, ran_at="2026-01-01T00:00:00Z",
                     result=RESULT, comparison={}, owner=owner)
    store.save_worksheet(create(
        worksheet_id=WORKSHEET, owner_id=owner, scenario_ref=PLAN,
        primary_run_ref=RUN, created_at="2026-01-01T00:00:00Z"))
    store.record_planned_event(
        owner=owner, worksheet_id=WORKSHEET,
        event=PlannedEvent(event_id="pe-shared", grant_ref="grant/g1",
                           expected_date="2026-06-15", employer_asset="ACME",
                           expected_gross_shares="152.26"),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1")
    store.record_observed_event(
        owner=owner, worksheet_id=WORKSHEET,
        event=ObservedEvent(observation_id="oe-shared",
                            observed_date="2026-06-16",
                            effective_date="2026-06-15", grant_ref="grant/g1",
                            employer_asset="ACME", gross_shares="152.26"),
        created_at="2026-01-01T00:00:00Z")
    store.record_reconciliation(
        owner=owner, worksheet_id=WORKSHEET,
        reconciliation=EventReconciliation(
            reconciliation_id="rc-shared",
            status=ReconciliationStatus.MATCHED, planned_ref="pe-shared",
            observed_ref="oe-shared", derived_at="2026-06-17T00:00:00Z"))
    # A worksheet proposal, so the indirect fixture has a parent to hang from.
    store.save_worksheet_proposal(
        proposal_id=PROPOSAL, owner=owner, worksheet_id=WORKSHEET,
        proposal=_Proposal(), created_at="2026-01-01T00:00:00Z")
    ownership_fixture.add(store, child_id=f"{CHILD}-{owner}",
                          proposal_id=PROPOSAL, proposal_owner=owner)
    return store


class _Proposal:
    source_revision = 1

    def to_json(self):
        return {"body": "a reviewed diff"}


@pytest.fixture
def tenants(monkeypatch):
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    database.create_all()

    store = session()
    ownership_fixture.create(store)

    # The indirect fixture joins the classified inventory for these tests only.
    # It is not in the migrations or the production registry — the retention
    # inventory test reads the schema and would fail on an unclassified table.
    extended = dict(WORKSPACE_RECORDS)
    extended[ownership_fixture.TABLE] = ownership_fixture.INDIRECT_CHILD
    monkeypatch.setattr(retention, "WORKSPACE_RECORDS", extended)

    seed(A)
    seed(B)
    return store


class TestTheDeclaredOrderSucceeds:
    def test_every_row_of_one_tenant_goes(self, tenants):
        receipt = delete_workspace(session(), A,
                                   requested_at="2026-07-01T00:00:00Z")
        assert receipt.status == "COMPLETE"
        assert verify_deleted(session(), A) == {}

    def test_the_indirect_fixture_is_removed_through_its_declared_path(
            self, tenants):
        """It has no owner column. A deletion written around `WHERE owner = ?`
        removes nothing from it and reports success."""
        assert ownership_fixture.INDIRECT_CHILD.owner_scope is OwnerScope.INDIRECT
        before = observe(
            f"SELECT child_id FROM {ownership_fixture.TABLE} ORDER BY child_id")
        assert len(before) == 2

        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")

        after = observe(
            f"SELECT child_id FROM {ownership_fixture.TABLE} ORDER BY child_id")
        assert [r["child_id"] for r in after] == [f"{CHILD}-{B}"]

    def test_the_other_tenant_is_byte_for_byte_unchanged(self, tenants):
        before = {
            "worksheet": observe("SELECT canonical_hash FROM worksheet "
                                 "WHERE owner = %s ORDER BY revision", (B,)),
            "plan": observe("SELECT content_hash FROM plan WHERE owner = %s",
                            (B,)),
            "plan_run": observe("SELECT run_id FROM plan_run WHERE owner = %s",
                                (B,)),
            "observed": observe("SELECT content_hash FROM observed_event "
                                "WHERE owner = %s", (B,)),
        }
        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        assert before["worksheet"] == observe(
            "SELECT canonical_hash FROM worksheet WHERE owner = %s "
            "ORDER BY revision", (B,))
        assert before["plan"] == observe(
            "SELECT content_hash FROM plan WHERE owner = %s", (B,))
        assert before["plan_run"] == observe(
            "SELECT run_id FROM plan_run WHERE owner = %s", (B,))
        assert before["observed"] == observe(
            "SELECT content_hash FROM observed_event WHERE owner = %s", (B,))

    def test_verified_from_a_fresh_session_after_commit(self, tenants):
        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        for record in WORKSPACE_RECORDS.values():
            if record.owner_scope is not OwnerScope.DIRECT:
                continue
            assert observe(
                f"SELECT COUNT(*) AS n FROM {record.table} "
                f"WHERE {record.owner_column} = %s", (A,))[0]["n"] == 0, \
                record.table


class TestTheDatabaseRefusesTheWrongOrder:
    def test_parent_first_is_refused(self, tenants):
        """`plan` before `plan_run` — the constraint is what makes the
        application's ordering a guarantee rather than a habit."""
        with pytest.raises(Exception) as caught:
            with session()._conn() as conn:
                conn.execute("DELETE FROM plan WHERE owner = ?", (A,))
        assert "foreign key" in str(caught.value).lower() or \
               "violates" in str(caught.value).lower()

    def test_the_refusal_leaves_everything_in_place(self, tenants):
        with pytest.raises(Exception):
            with session()._conn() as conn:
                conn.execute("DELETE FROM plan WHERE owner = ?", (A,))
        assert observe("SELECT COUNT(*) AS n FROM plan WHERE owner = %s",
                       (A,))[0]["n"] == 1

    def test_events_before_their_reconciliation_is_refused(self, tenants):
        with pytest.raises(Exception) as caught:
            with session()._conn() as conn:
                conn.execute("DELETE FROM observed_event WHERE owner = ?", (A,))
        assert "foreign key" in str(caught.value).lower() or \
               "violates" in str(caught.value).lower()


class TestVerificationIsIndependentOfTheDeletion:
    def test_omitting_a_classified_table_fails_verification(self, tenants,
                                                            monkeypatch):
        """The deletion is made incomplete *below* the code that checks it.

        Verification reads the classified inventory rather than the deletion
        that just ran, so a table the deletion skipped is still enumerated.
        Sharing that list would make the two agree by construction.
        """
        from src.workspace import erasure

        skipped = "confirmation_event"
        # Captured before patching. Calling the patched name inside its own
        # replacement recursed until the stack ran out, which looked like a
        # test failure and was a test bug.
        full = erasure.owner_scoped_tables

        def partial():
            return tuple(one for one in full() if one.table != skipped)

        session().record_confirmation_event(
            event_id="ce-1", owner=A, occurred_at="2026-01-01T00:00:00Z",
            kind="EDIT", path=None, field=None, provenance=None,
            original_value=None, final_value=None, reason=None,
            compiler_version=None, defaults_ref=None)

        monkeypatch.setattr(erasure, "owner_scoped_tables", partial)
        with pytest.raises(DeletionIncomplete) as caught:
            delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        monkeypatch.undo()
        assert skipped in str(caught.value)

    def test_verification_reports_which_table_survived(self, tenants,
                                                       monkeypatch):
        from src.workspace import erasure

        session().record_confirmation_event(
            event_id="ce-1", owner=A, occurred_at="2026-01-01T00:00:00Z",
            kind="EDIT", path=None, field=None, provenance=None,
            original_value=None, final_value=None, reason=None,
            compiler_version=None, defaults_ref=None)
        monkeypatch.setattr(
            erasure, "owner_scoped_tables",
            lambda: tuple(one for one in WORKSPACE_RECORDS.values()
                          if one.table != "confirmation_event"
                          and one.deletion_behaviour.value == "DELETE_WITH_OWNER"))
        with pytest.raises(DeletionIncomplete):
            delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        monkeypatch.undo()
        remaining = verify_deleted(session(), A)
        assert "confirmation_event" in remaining


class TestAWrongJoinIsCaught:
    """The mutation that still returns rows, and returns the wrong tenant's.

    A missing table is easy to notice. An ownership path that joins on less
    than its parent's key is not: the deletion runs, reports plausible counts,
    and removes another tenant's rows. No deletion test catches that by
    itself — under the broken path Alice's deletion simply takes Bob's child
    with it and nothing looks wrong.

    So the protection is mechanical and sits before the deletion: a declared
    path must span its parent's whole primary key, checked against the keys
    PostgreSQL reports rather than against another declaration.
    """

    def parent_keys(self):
        from sqlalchemy import inspect

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        try:
            inspector = inspect(engine)
            return {name: inspector.get_pk_constraint(name)["constrained_columns"]
                    for name in inspector.get_table_names()}
        finally:
            engine.dispose()

    def test_every_declared_path_spans_its_parents_key(self, tenants):
        wrong = retention.paths_not_spanning_their_parent_key(self.parent_keys())
        assert wrong == (), f"ownership paths that under-join: {wrong}"

    def test_a_truncated_path_is_rejected(self, tenants, monkeypatch):
        """Dropping the tenant column from the join must be caught here."""
        import dataclasses

        broken = dict(retention.WORKSPACE_RECORDS)
        broken[ownership_fixture.TABLE] = dataclasses.replace(
            broken[ownership_fixture.TABLE],
            ownership_path=OwnershipPath(
                local_key="proposal_id", parent_table="worksheet_proposal",
                parent_key="proposal_id", parent_owner_column="owner"))
        monkeypatch.setattr(retention, "WORKSPACE_RECORDS", broken)

        wrong = retention.paths_not_spanning_their_parent_key(self.parent_keys())
        assert wrong, "a path joining on half its parent's key was accepted"
        assert ownership_fixture.TABLE in wrong[0]

    def test_the_truncated_path_really_would_cross_tenants(self, tenants,
                                                           monkeypatch):
        """Why the guard is needed, shown rather than asserted.

        With the tenant column dropped, deleting Alice takes Bob's child row
        too — and the deletion reports success either way.
        """
        import dataclasses

        broken = dict(retention.WORKSPACE_RECORDS)
        broken[ownership_fixture.TABLE] = dataclasses.replace(
            broken[ownership_fixture.TABLE],
            ownership_path=OwnershipPath(
                local_key="proposal_id", parent_table="worksheet_proposal",
                parent_key="proposal_id", parent_owner_column="owner"))
        monkeypatch.setattr(retention, "WORKSPACE_RECORDS", broken)

        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        survivors = {r["child_id"] for r in observe(
            f"SELECT child_id FROM {ownership_fixture.TABLE}")}
        assert survivors == set(), (
            "expected the truncated join to have taken both tenants' rows; if "
            "it did not, this test no longer demonstrates the hazard")


class TestTheReceipt:
    def test_it_carries_counts_and_policy_not_content(self, tenants):
        receipt = delete_workspace(session(), A,
                                   requested_at="2026-07-01T00:00:00Z")
        body = str(receipt.to_json())
        assert receipt.policy_version
        assert receipt.counts
        for secret in ("152.26", "ACME", "grant/g1", "a reviewed diff",
                       "Roth IRA", A, B):
            assert secret not in body, f"{secret!r} appeared in a receipt"

    def test_the_owner_reference_is_irreversible(self, tenants):
        receipt = delete_workspace(session(), A,
                                   requested_at="2026-07-01T00:00:00Z")
        assert receipt.owner_reference.startswith("owner-")
        assert A not in receipt.owner_reference

    def test_counts_name_the_tables_that_were_emptied(self, tenants):
        receipt = delete_workspace(session(), A,
                                   requested_at="2026-07-01T00:00:00Z")
        assert receipt.counts["worksheet"] == 1
        assert receipt.counts["plan_run"] == 1
        assert receipt.counts[ownership_fixture.TABLE] == 1


class TestExportSeesWhatDeletionRemoves:
    def test_the_indirect_fixture_is_exported_too(self, tenants):
        """An export missing something a deletion removes is a user who could
        not see what they lost."""
        payload = export_workspace(session(), A)
        assert payload["counts"][ownership_fixture.TABLE] == 1

    def test_the_export_holds_only_its_own_tenant(self, tenants):
        payload = export_workspace(session(), A)
        assert f"{CHILD}-{B}" not in str(payload)
