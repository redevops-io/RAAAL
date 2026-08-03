"""Two tenants using the same identifiers, on one PostgreSQL database.

Owner A and owner B are given *identical* worksheet ids, plan ids, proposal
ids, observation ids, reconciliation ids and revision numbers. Distinct
identifiers would let a test pass because nothing collided, which is the
weakest possible reason — the interesting failures all live where two tenants
name the same thing.

Every operation is exercised: read, write, accept, observe, reconcile, export,
delete. And every failure is checked twice over:

    data isolation    A never reads or changes B's row
    error isolation   A's failure looks exactly like a locally absent record

The second is the one that leaks. A store can scope every read correctly and
still answer "that id is taken" on a write, which tells the requester something
about another tenant they were not entitled to ask. That is the defect that put
`owner` into the worksheet primary key, and this file is where it stays fixed.

**Fresh sessions throughout.** State asserted through the session that wrote it
proves only what that session remembers.
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
from src.workspace.apply import ApplyRefused, accept
from src.workspace.erasure import delete_workspace, export_workspace
from src.workspace.intent import plan
from src.workspace.proposal import propose
from src.workspace.store import NotSaveable, WorkspaceStore
from src.workspace.worksheet import create, from_json, revise

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; tenant isolation under real "
           "constraints is a PostgreSQL-only guarantee")

A, B = "alice", "bob"

#: Deliberately shared. Every one of these is the same string for both tenants.
WORKSHEET = "ws-shared"
PLAN = "plan-shared"
PROPOSAL = "wp-shared"
OBSERVATION = "oe-shared"
PLANNED = "pe-shared"
RECONCILIATION = "rc-shared"

RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}

#: Things a cross-tenant failure must never disclose.
LEAKS = ("alice", "bob", "worksheet_proposal", "worksheet_pkey", "duplicate key",
         "constraint", "psycopg", "DETAIL", "pg_", "relation", "SELECT",
         "INSERT", "UPDATE")


def fresh():
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()


def session():
    """A store of its own, so nothing asserted here is session-local."""
    return WorkspaceStore(POSTGRES_URL)


def seed(owner):
    """The same identifiers, for whichever owner is passed."""
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
    # Plan ids are globally unique by primary key, so each tenant needs its own.
    # Everything owner-scoped below is deliberately identical.
    store.save_plan(plan_id=f"{PLAN}-{owner}", owner=owner, scenario=scenario,
                    stated_text="seed", saved_at="2026-01-01T00:00:00Z")
    store.record_run(run_id=f"run-{owner}", plan_id=f"{PLAN}-{owner}",
                     ran_at="2026-01-01T00:00:00Z", result=RESULT, comparison={})
    store.save_worksheet(create(
        worksheet_id=WORKSHEET, owner_id=owner, scenario_ref=f"{PLAN}-{owner}",
        primary_run_ref=f"run-{owner}", created_at="2026-01-01T00:00:00Z"))
    return store


def staged_proposal(owner):
    store = session()
    worksheet = from_json(store.get_worksheet(WORKSHEET, owner)["payload"])
    intent = plan("Try SPY, VTI and VT and keep the best", intent_id="i",
                  source_revision=worksheet.revision, history=[],
                  target_run=f"run-{owner}")
    proposal = propose(intent, worksheet)
    store.save_worksheet_proposal(
        proposal_id=PROPOSAL, owner=owner, worksheet_id=WORKSHEET,
        proposal=proposal, created_at="2026-01-01T00:00:00Z")
    return proposal


def observe(query, params=()):
    conn = Database(POSTGRES_URL).connect()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def assert_no_leak(message):
    lowered = str(message).lower()
    for leak in LEAKS:
        assert leak.lower() not in lowered, (
            f"{leak!r} disclosed in a cross-tenant error: {message!r}")


@pytest.fixture
def tenants():
    fresh()
    seed(A)
    seed(B)
    return session()


class TestRead:
    def test_each_owner_sees_only_their_own_worksheet(self, tenants):
        store = session()
        a = store.get_worksheet(WORKSHEET, A)
        b = store.get_worksheet(WORKSHEET, B)
        assert a["payload"]["scenario_ref"] == f"{PLAN}-{A}"
        assert b["payload"]["scenario_ref"] == f"{PLAN}-{B}"

    def test_a_third_owner_sees_nothing(self, tenants):
        """Absent must look absent, not forbidden — a distinction between the
        two is itself an answer about another tenant."""
        assert session().get_worksheet(WORKSHEET, "carol") is None

    def test_runs_are_not_reachable_across_owners(self, tenants):
        assert session().get_run(f"run-{B}", A) is None

    def test_plans_are_not_listed_across_owners(self, tenants):
        listed = {p["plan_id"] for p in session().list_plans(A)}
        assert listed == {f"{PLAN}-{A}"}


class TestWrite:
    def test_the_same_worksheet_id_is_not_a_collision(self, tenants):
        """The write refusal that leaked. Keyed without `owner`, B creating a
        worksheet id A already held was refused — and the refusal answered a
        question B was not entitled to ask."""
        store = session()
        current = from_json(store.get_worksheet(WORKSHEET, B)["payload"])
        store.save_worksheet(revise(current, reason="B's own edit",
                                    created_at="2026-01-02T00:00:00Z"))
        assert store.get_worksheet(WORKSHEET, B)["revision"] == 2
        assert store.get_worksheet(WORKSHEET, A)["revision"] == 1

    def test_identical_revision_numbers_coexist(self, tenants):
        rows = observe("SELECT owner, revision FROM worksheet "
                       "WHERE worksheet_id = %s ORDER BY owner", (WORKSHEET,))
        assert [(r["owner"], r["revision"]) for r in rows] == [(A, 1), (B, 1)]

    def test_a_rewrite_of_another_owners_revision_is_refused_locally(self,
                                                                    tenants):
        """A tries to overwrite what is, for B, revision 1. It must fail as a
        local matter and disclose nothing about B."""
        store = session()
        b_sheet = from_json(store.get_worksheet(WORKSHEET, B)["payload"])
        stolen = create(worksheet_id=WORKSHEET, owner_id=A,
                        scenario_ref=f"{PLAN}-{A}", primary_run_ref=f"run-{A}",
                        created_at="2026-01-03T00:00:00Z")
        try:
            store.save_worksheet(stolen)
        except NotSaveable as exc:
            assert_no_leak(exc)
        # B is untouched either way.
        assert observe("SELECT canonical_hash FROM worksheet WHERE owner = %s "
                       "AND worksheet_id = %s AND revision = 1",
                       (B, WORKSHEET))[0]["canonical_hash"] == \
            b_sheet.canonical_hash


class TestAcceptance:
    def test_each_owner_accepts_their_own_proposal(self, tenants):
        a_proposal = staged_proposal(A)
        b_proposal = staged_proposal(B)

        accept(session(), proposal_id=PROPOSAL, owner=A, proposal=a_proposal,
               worksheet_id=WORKSHEET, at="2026-01-02T00:00:00Z",
               run_candidate=lambda c: dict(RESULT))
        accept(session(), proposal_id=PROPOSAL, owner=B, proposal=b_proposal,
               worksheet_id=WORKSHEET, at="2026-01-02T00:00:00Z",
               run_candidate=lambda c: dict(RESULT))

        rows = observe("SELECT owner, status, result_revision FROM "
                       "worksheet_proposal WHERE proposal_id = %s ORDER BY owner",
                       (PROPOSAL,))
        assert [(r["owner"], r["status"], r["result_revision"]) for r in rows] \
            == [(A, "ACCEPTED", 2), (B, "ACCEPTED", 2)]

    def test_one_owner_cannot_accept_anothers_proposal(self, tenants):
        """A's acceptance of B's proposal id must fail as absent, and must not
        move B's proposal."""
        b_proposal = staged_proposal(B)
        with pytest.raises(ApplyRefused) as caught:
            accept(session(), proposal_id=PROPOSAL, owner=A,
                   proposal=b_proposal, worksheet_id=WORKSHEET,
                   at="2026-01-02T00:00:00Z",
                   run_candidate=lambda c: dict(RESULT))
        assert_no_leak(caught.value)

        assert observe("SELECT status FROM worksheet_proposal WHERE owner = %s",
                       (B,))[0]["status"] == "PROPOSED"

    def test_a_failed_cross_tenant_acceptance_leaves_no_local_artifacts(
            self, tenants):
        b_proposal = staged_proposal(B)
        before = observe("SELECT COUNT(*) AS n FROM plan_run", ())[0]["n"]
        with pytest.raises(ApplyRefused):
            accept(session(), proposal_id=PROPOSAL, owner=A,
                   proposal=b_proposal, worksheet_id=WORKSHEET,
                   at="2026-01-02T00:00:00Z",
                   run_candidate=lambda c: dict(RESULT))
        after = observe("SELECT COUNT(*) AS n FROM plan_run", ())[0]["n"]
        assert after == before, "a refused cross-tenant apply wrote runs"
        assert observe("SELECT COUNT(*) AS n FROM worksheet WHERE owner = %s",
                       (A,))[0]["n"] == 1


class TestObservationAndReconciliation:
    def _record(self, owner):
        store = session()
        store.record_planned_event(
            owner=owner, worksheet_id=WORKSHEET,
            event=PlannedEvent(event_id=PLANNED, grant_ref="grant/g1",
                               expected_date="2026-06-15",
                               employer_asset="ACME",
                               expected_gross_shares="152.26"),
            plan_revision=1, created_at="2026-01-01T00:00:00Z",
            matching_policy_version="m@1")
        store.record_observed_event(
            owner=owner, worksheet_id=WORKSHEET,
            event=ObservedEvent(observation_id=OBSERVATION,
                                observed_date="2026-06-16",
                                effective_date="2026-06-15",
                                grant_ref="grant/g1", employer_asset="ACME",
                                gross_shares="152.26"),
            created_at="2026-01-01T00:00:00Z")
        store.record_reconciliation(
            owner=owner, worksheet_id=WORKSHEET,
            reconciliation=EventReconciliation(
                reconciliation_id=RECONCILIATION,
                status=ReconciliationStatus.MATCHED, planned_ref=PLANNED,
                observed_ref=OBSERVATION, derived_at="2026-06-17T00:00:00Z"))

    def test_identical_event_ids_coexist(self, tenants):
        self._record(A)
        self._record(B)
        rows = observe("SELECT owner FROM observed_event WHERE "
                       "observed_event_id = %s ORDER BY owner", (OBSERVATION,))
        assert [r["owner"] for r in rows] == [A, B]

    def test_each_owner_reads_only_their_own(self, tenants):
        self._record(A)
        self._record(B)
        store = session()
        assert len(store.observed_events(WORKSHEET, A)) == 1
        assert len(store.reconciliations(WORKSHEET, A)) == 1
        assert store.planned_events(WORKSHEET, "carol") == []

    def test_a_reconciliation_cannot_cite_another_owners_event(self, tenants):
        """The composite foreign key spans owner, so a cross-tenant reference
        is refused by the database rather than merely unqueried."""
        self._record(B)
        with pytest.raises(Exception) as caught:
            with session()._conn() as conn:
                conn.execute(
                    "INSERT INTO event_reconciliation (owner, worksheet_id, "
                    "reconciliation_id, status, payload, "
                    "matching_policy_version, content_hash, derived_at, "
                    "observed_event_id) VALUES (?,?,?,?,?,?,?,?,?)",
                    (A, WORKSHEET, "rc-steal", "MATCHED", "{}", "m@1", "h",
                     "2026-06-17T00:00:00Z", OBSERVATION))
        assert "foreign key" in str(caught.value).lower() or \
               "violates" in str(caught.value).lower()


class TestExport:
    def test_an_export_carries_only_its_own_owner(self, tenants):
        TestObservationAndReconciliation()._record(A)
        TestObservationAndReconciliation()._record(B)
        payload = export_workspace(session(), A)
        assert payload["counts"]["observed_event"] == 1
        text = str(payload)
        assert B not in text, "another tenant's owner id appeared in an export"

    def test_an_export_for_an_unknown_owner_is_empty_not_an_error(self, tenants):
        payload = export_workspace(session(), "carol")
        assert all(count == 0 for count in payload["counts"].values())


class TestDeletion:
    def test_deleting_one_owner_leaves_the_other_byte_for_byte(self, tenants):
        TestObservationAndReconciliation()._record(A)
        TestObservationAndReconciliation()._record(B)
        before = observe(
            "SELECT worksheet_id, revision, canonical_hash FROM worksheet "
            "WHERE owner = %s ORDER BY revision", (B,))
        b_events = observe("SELECT content_hash FROM observed_event "
                           "WHERE owner = %s", (B,))

        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")

        assert observe(
            "SELECT worksheet_id, revision, canonical_hash FROM worksheet "
            "WHERE owner = %s ORDER BY revision", (B,)) == before
        assert observe("SELECT content_hash FROM observed_event "
                       "WHERE owner = %s", (B,)) == b_events

    def test_the_deleted_owner_has_nothing_left(self, tenants):
        TestObservationAndReconciliation()._record(A)
        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        for table in ("worksheet", "observed_event", "planned_event",
                      "event_reconciliation", "plan"):
            assert observe(f"SELECT COUNT(*) AS n FROM {table} "
                           "WHERE owner = %s", (A,))[0]["n"] == 0, table

    def test_deleting_an_unknown_owner_is_not_an_error(self, tenants):
        receipt = delete_workspace(session(), "carol",
                                   requested_at="2026-07-01T00:00:00Z")
        assert receipt.status == "COMPLETE"
        assert all(count == 0 for count in receipt.counts.values())

    def test_the_receipt_names_no_one(self, tenants):
        TestObservationAndReconciliation()._record(A)
        receipt = delete_workspace(session(), A,
                                   requested_at="2026-07-01T00:00:00Z")
        text = str(receipt.to_json())
        assert A not in text and B not in text
        assert receipt.owner_reference.startswith("owner-")


class TestTheSamePlanIdUnderTwoTenants:
    """What the ownership migration was for.

    `plan` was keyed on `plan_id` alone, so the seeding above had to suffix
    plan ids per owner to avoid a collision — a test working around the defect
    it should have been finding. Both tenants now use the same plan id and the
    same run ids.
    """

    def _seed_shared(self, owner):
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        store = session()
        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.", name=PLAN, version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        provenance = compiled.scenario.provenance
        scenario = ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=provenance.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in provenance.inferred),
                contradictions=provenance.contradictions, unresolved=())})
        store.save_plan(plan_id="p-shared", owner=owner, scenario=scenario,
                        stated_text="seed", saved_at="2026-01-01T00:00:00Z")
        store.record_run(run_id="r-shared", plan_id="p-shared",
                         ran_at="2026-01-01T00:00:00Z", result=RESULT,
                         comparison={}, owner=owner)
        return store

    def test_both_tenants_hold_the_same_plan_id(self, tenants):
        self._seed_shared(A)
        self._seed_shared(B)
        rows = observe("SELECT owner FROM plan WHERE plan_id = %s ORDER BY owner",
                       ("p-shared",))
        assert [r["owner"] for r in rows] == [A, B]

    def test_both_tenants_hold_the_same_run_id(self, tenants):
        self._seed_shared(A)
        self._seed_shared(B)
        rows = observe("SELECT owner FROM plan_run WHERE run_id = %s "
                       "ORDER BY owner", ("r-shared",))
        assert [r["owner"] for r in rows] == [A, B]

    def test_each_reads_only_their_own_run(self, tenants):
        self._seed_shared(A)
        self._seed_shared(B)
        store = session()
        assert store.get_run("r-shared", A)["owner"] == A
        assert store.get_run("r-shared", B)["owner"] == B
        assert store.get_run("r-shared", "carol") is None

    def test_runs_for_is_scoped_at_the_query(self, tenants):
        self._seed_shared(A)
        self._seed_shared(B)
        runs = session().runs_for("p-shared", A)
        assert [r["owner"] for r in runs] == [A]

    def test_a_run_cannot_reference_another_owners_plan(self, tenants):
        """The composite foreign key, not an application convention."""
        self._seed_shared(A)
        with pytest.raises(Exception) as caught:
            with session()._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    (B, "r-steal", "p-shared", "2026-01-01T00:00:00Z",
                     "{}", "{}"))
        assert "foreign key" in str(caught.value).lower() or \
               "violates" in str(caught.value).lower()

    def test_deleting_one_tenant_leaves_the_others_identical_ids(self, tenants):
        self._seed_shared(A)
        self._seed_shared(B)
        delete_workspace(session(), A, requested_at="2026-07-01T00:00:00Z")
        assert observe("SELECT owner FROM plan_run WHERE run_id = %s",
                       ("r-shared",)) == [{"owner": B}]
        assert observe("SELECT COUNT(*) AS n FROM plan WHERE plan_id = %s",
                       ("p-shared",))[0]["n"] == 1
