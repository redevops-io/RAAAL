"""One invariant, checked at three layers, against three different sources.

Five tables once let one tenant overwrite another's row. They were found one at
a time and fixed one at a time, and they were never five bugs — they were one
rule violated in five places with nothing enumerating the rule. Then a correct
key migration broke a correct consumer, because `OwnershipPath` still joined on
the scalar identity those keys used to have.

    schema      PostgreSQL metadata        what the database permits
    writes      captured adapted SQL       what writers ask it to do
    consumers   captured adapted SQL       whether a join keeps the identity
    behaviour   Alice and Bob, same ids    whether it actually crosses

Each catches an omission the others cannot see, and the behavioural fixtures
are what stop all three passing on a technicality.

The schema check reads PostgreSQL rather than `src/db/schema.py` on purpose: a
model and its migrations can share an omission, and the migration-parity test
would agree with both.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import os

import pytest

from src.db import tenancy
from src.db.engine import Database, capture_statements
from src.db.tenancy import (
    consumer_violations,
    schema_violations,
    tenant_owned_tables,
    write_violations,
)
from src.mission.rsu_reconcile import (
    EventReconciliation,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
)
from src.workspace.erasure import delete_workspace, export_workspace
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; the schema layer reads PostgreSQL "
           "metadata and the write layer reads adapted statements")

A, B = "alice", "bob"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}

#: Identical across both tenants, so a collision is possible at every layer.
PLAN, RUN, WORKSHEET = "p-shared", "r-shared", "ws-shared"


def scenario_for(name=PLAN):
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name=name, version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    return ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})


@pytest.fixture(scope="module")
def database():
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    database.create_all()
    return database


@pytest.fixture(scope="module")
def metadata(database):
    """Read from PostgreSQL, not from the model."""
    from sqlalchemy import inspect

    engine = database.sqlalchemy_engine()
    try:
        inspector = inspect(engine)
        names = inspector.get_table_names()
        return {
            "columns": {n: inspector.get_columns(n) for n in names},
            "primary_keys": {
                n: inspector.get_pk_constraint(n)["constrained_columns"]
                for n in names},
            "unique": {n: [u["column_names"]
                           for u in inspector.get_unique_constraints(n)]
                       + [i["column_names"] for i in inspector.get_indexes(n)
                          if i.get("unique")]
                       for n in names},
            "foreign_keys": {n: inspector.get_foreign_keys(n) for n in names},
        }
    finally:
        engine.dispose()


#: Distinguishes repeated passes, so an immutable row is not rewritten.
_PASS = {"n": 0}


def every_write(store, owner):
    """Call every writing store method, and report which ones were called.

    The names are returned by the same structure that performs the calls, so
    the coverage claim cannot drift from the coverage. A hand-written list of
    "methods exercised" passed while `every_write` called none of them — it
    asserted its own claim, which is the weakest possible check.
    """
    from src.workspace.intent import plan as plan_intent
    from src.workspace.proposal import propose
    from src.workspace.worksheet import from_json

    called = []
    _PASS["n"] += 1
    suffix = f"{owner}-{_PASS['n']}"

    def call(name, thunk):
        called.append(name)
        return thunk()

    call("save_plan", lambda: store.save_plan(
        plan_id=PLAN, owner=owner, scenario=scenario_for(), stated_text="x",
        saved_at="2026-01-01T00:00:00Z"))
    # The delivery before the run that cites it, as production does — so the
    # tenancy capture sees the real pairing rather than a run with no evidence.
    access = _access_for(owner, RUN)
    call("record_access_event", lambda: store.record_access_event(
        access.access_event, owner=owner))
    call("record_run", lambda: store.record_run(
        run_id=RUN, plan_id=PLAN, ran_at="2026-01-01T00:00:00Z",
        result={**RESULT, "market_data": access.provenance.to_json()},
        comparison={}, owner=owner,
        access_event_id=access.access_event_id))
    call("save_worksheet", lambda: store.save_worksheet(create(
        worksheet_id=WORKSHEET, owner_id=owner, scenario_ref=PLAN,
        primary_run_ref=RUN, created_at="2026-01-01T00:00:00Z")))
    call("record_planned_event", lambda: store.record_planned_event(
        owner=owner, worksheet_id=WORKSHEET,
        event=PlannedEvent(event_id="pe-1", grant_ref="g",
                           expected_date="2026-06-15",
                           expected_gross_shares="152.26"),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1"))
    call("record_observed_event", lambda: store.record_observed_event(
        owner=owner, worksheet_id=WORKSHEET,
        event=ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                            effective_date="2026-06-15",
                            gross_shares="152.26"),
        created_at="2026-01-01T00:00:00Z"))
    call("record_reconciliation", lambda: store.record_reconciliation(
        owner=owner, worksheet_id=WORKSHEET,
        reconciliation=EventReconciliation(
            reconciliation_id="rc-1", status=ReconciliationStatus.MATCHED,
            planned_ref="pe-1", observed_ref="oe-1",
            derived_at="2026-06-17T00:00:00Z")))
    call("record_confirmation_event", lambda: store.record_confirmation_event(
        event_id="ce-1", owner=owner, occurred_at="2026-01-01T00:00:00Z",
        kind="EDIT", path=None, field=None, provenance=None,
        original_value=None, final_value=None, reason=None,
        compiler_version=None, defaults_ref=None))

    worksheet = from_json(store.get_worksheet(WORKSHEET, owner)["payload"])
    intent = plan_intent("show me the drawdown as a chart",
                         intent_id=f"i-{suffix}",
                         source_revision=worksheet.revision, history=[],
                         target_run=RUN)
    call("append_worksheet_intent", lambda: store.append_worksheet_intent(
        worksheet_id=WORKSHEET, owner=owner, intent=intent,
        created_at="2026-01-01T00:00:00Z", planner_version="planner@1",
        instruction_hash="h"))
    call("save_worksheet_proposal", lambda: store.save_worksheet_proposal(
        proposal_id=f"wp-{suffix}", owner=owner, worksheet_id=WORKSHEET,
        proposal=propose(intent, worksheet),
        created_at="2026-01-01T00:00:00Z"))
    call("link_intent_proposal", lambda: store.link_intent_proposal(
        intent.intent_id, owner, proposal_id=f"wp-{suffix}"))
    call("resolve_worksheet_proposal",
         lambda: store.resolve_worksheet_proposal(
             f"wp-{suffix}", owner, status="REJECTED",
             resolved_at="2026-01-02T00:00:00Z", actor="pilot"))
    call("save_proposal", lambda: store.save_proposal(
        owner=owner, proposal=_MissionProposal()))
    call("save_observation", lambda: store.save_observation(
        owner=owner, observation=_MissionObservation()))
    return called


#: Public store methods that only read. Everything else must be exercised by
#: `every_write`, which is what stops a writer being missed by omission — the
#: write layer is blind to a statement no test issues, and hand-picking the
#: methods to exercise is how `link_intent_proposal` escaped it.
READ_ONLY = {
    "get_plan", "get_run", "get_worksheet", "get_worksheet_proposal",
    "list_plans", "list_proposals", "list_observations", "runs_for",
    "planned_events", "observed_events", "reconciliations",
    "worksheet_intents", "worksheet_revisions", "worksheet_for_scenario",
    "confirmation_events", "transaction", "rsu_context_of",
    "lock_worksheet_proposal",
    # Reads a stored run and the delivery it cites, and compares them. It
    # writes nothing — but it is owner-scoped, so `every_read` exercises it
    # and the read layer checks the scoping.
    "get_access_event", "verify_access_chain",
}


def write_methods():
    """Every public store method that is not declared read-only."""
    return {name for name in dir(WorkspaceStore)
            if not name.startswith("_") and callable(getattr(WorkspaceStore, name))
            and name not in READ_ONLY}


#: Deliberately shared between tenants, like `PLAN` and `RUN` above. This file
#: exists to force identifier collision: two owners holding the same id is the
#: state under which an unscoped write silently overwrites the other's row, and
#: an isolation failure that needs distinct ids to appear is one nobody would
#: hit in the field either.
ACCESS_EVENT = "mdae-shared"


def _access_for(owner, run_id):
    """A delivery with a fixed identity and a fixed instant.

    Stable across passes on purpose. `every_write` runs several times per
    owner, and a fresh event id each pass would change the run body and turn
    the second write into a conflict — the store refusing correctly, and the
    tenancy capture never reaching the statements it exists to inspect.
    """
    import dataclasses

    from src.market_data.access import resolve

    access = resolve(context="tenancy capture", run_id=run_id,
                     accessed_at="2026-01-01T00:00:00Z",
                     request_id="req-shared")
    return dataclasses.replace(
        access, access_event=dataclasses.replace(
            access.access_event, access_event_id=ACCESS_EVENT))


def every_read(store, owner):
    store.get_worksheet(WORKSHEET, owner)
    store.get_run(RUN, owner)
    store.get_run(RUN, owner)
    store.runs_for(PLAN, owner)
    store.list_plans(owner)
    store.get_plan(PLAN, owner)
    store.planned_events(WORKSHEET, owner)
    store.observed_events(WORKSHEET, owner)
    store.reconciliations(WORKSHEET, owner)
    store.worksheet_intents(WORKSHEET, owner)
    store.confirmation_events(owner)
    store.verify_access_chain(RUN, owner)
    export_workspace(store, owner)


class _MissionProposal:
    """Minimal stand-in for the forward-tracking proposal artifact."""

    artifact_id = "mp-1"
    proposal_id = "mp-1"
    plan_id = PLAN
    generated_at = "2026-01-01T00:00:00Z"

    class _Status:
        value = "OPEN"

    status = _Status()

    def to_json(self):
        return {"body": "a forward-tracking proposal"}


class _MissionObservation:
    artifact_id = "mo-1"
    plan_id = PLAN
    observed_at = "2026-01-01T00:00:00Z"

    def to_json(self):
        return {"body": "an observation"}


class TestTheSchemaLayer:
    """What the deployed database permits."""

    def test_no_tenant_owned_table_is_unsafe(self, metadata):
        found = schema_violations(
            metadata["columns"], metadata["primary_keys"], metadata["unique"],
            metadata["foreign_keys"], tenant_owned_tables())
        assert found == [], "\n".join(str(one) for one in found)

    def test_the_exception_list_is_empty(self, metadata):
        """`plan` was the last exception and the migration closed it."""
        unscoped = [table for table in tenant_owned_tables()
                    if "owner" not in metadata["primary_keys"].get(table, ())]
        assert unscoped == []

    def test_the_check_reads_postgresql_not_the_model(self, metadata):
        """A model and its migrations can share an omission, and the parity
        test would agree with both."""
        assert metadata["primary_keys"]["plan"], "no metadata was read"
        assert set(metadata["columns"]) >= set(tenant_owned_tables())

    def test_it_detects_a_key_without_owner(self, metadata):
        broken = dict(metadata["primary_keys"])
        broken["worksheet_proposal"] = ["proposal_id"]
        found = schema_violations(
            metadata["columns"], broken, metadata["unique"],
            metadata["foreign_keys"], tenant_owned_tables())
        assert any("omits `owner`" in one.detail for one in found)

    def test_it_detects_a_nullable_owner(self, metadata):
        broken = {name: [dict(c, nullable=True) if c["name"] == "owner" else c
                         for c in columns]
                  for name, columns in metadata["columns"].items()}
        found = schema_violations(
            broken, metadata["primary_keys"], metadata["unique"],
            metadata["foreign_keys"], tenant_owned_tables())
        assert any("nullable" in one.detail for one in found)

    def test_it_detects_a_foreign_key_without_owner(self, metadata):
        broken = dict(metadata["foreign_keys"])
        broken["plan_run"] = [{"constrained_columns": ["plan_id"],
                               "referred_table": "plan",
                               "referred_columns": ["plan_id"]}]
        found = schema_violations(
            metadata["columns"], metadata["primary_keys"], metadata["unique"],
            broken, tenant_owned_tables())
        assert any("foreign key" in one.detail for one in found)


class TestTheWriteLayer:
    """What production writers ask the schema to do."""

    def test_every_write_supplies_and_scopes_by_owner(self, database):
        store = WorkspaceStore(POSTGRES_URL)
        with capture_statements() as issued:
            every_write(store, A)
        found = write_violations(issued, tenant_owned_tables())
        assert found == [], "\n".join(str(one) for one in found)

    def test_the_acceptance_path_is_covered_too(self, database):
        from src.workspace.apply import accept
        from src.workspace.intent import plan
        from src.workspace.proposal import propose
        from src.workspace.worksheet import from_json

        store = WorkspaceStore(POSTGRES_URL)
        worksheet = from_json(store.get_worksheet(WORKSHEET, A)["payload"])
        intent = plan("Try SPY, VTI and VT and keep the best", intent_id="i",
                      source_revision=worksheet.revision, history=[],
                      target_run=RUN)
        proposal = propose(intent, worksheet)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=A, worksheet_id=WORKSHEET,
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        with capture_statements() as issued:
            accept(store, proposal_id="wp-1", owner=A, proposal=proposal,
                   worksheet_id=WORKSHEET, at="2026-01-02T00:00:00Z",
                   run_candidate=lambda c: dict(RESULT))
        found = write_violations(issued, tenant_owned_tables())
        assert found == [], "\n".join(str(one) for one in found)

    def test_it_detects_a_conflict_target_without_owner(self):
        """The five original defects, in the form the statement shows."""
        found = write_violations(
            ["INSERT INTO worksheet_proposal (proposal_id, owner, payload) "
             "VALUES (%s,%s,%s) ON CONFLICT (proposal_id) DO UPDATE SET "
             "payload = EXCLUDED.payload"],
            tenant_owned_tables())
        assert any("conflict target" in one.detail for one in found)

    def test_it_detects_an_insert_without_owner(self):
        found = write_violations(
            ["INSERT INTO planned_event (worksheet_id, planned_event_id) "
             "VALUES (%s,%s)"], tenant_owned_tables())
        assert any("does not supply" in one.detail for one in found)

    def test_every_write_method_is_exercised(self):
        """The inventory, so the layer cannot be blind by omission.

        Falsifying an unscoped `UPDATE` in `link_intent_proposal` changed no
        result, because the capture never called it. A checker that never sees
        a statement reports it clean.
        """
        exercised = set(every_write(WorkspaceStore(POSTGRES_URL), "coverage"))
        missing = write_methods() - exercised
        assert missing == set(), (
            f"store methods that write and are never captured: {missing}. "
            "Add them to `every_write` or to READ_ONLY.")

    def test_it_detects_an_unscoped_update(self):
        found = write_violations(
            ["UPDATE worksheet_proposal SET status = %s WHERE proposal_id = %s"],
            tenant_owned_tables())
        assert any("not scoped" in one.detail for one in found)


class TestTheConsumerLayer:
    """Whether a join preserves the identity it references.

    The layer the `OwnershipPath` defect needed. A composite key is no
    protection if the consumer matches on the scalar identifier the key used to
    be — the query stays valid and returns another tenant's rows.

    **These tests assert they saw a join.** The first version did not, and
    passed on nothing: the ownership migration removed the store's only
    ownership join (`get_run` used to reach the owner through `plan`), and with
    no indirect tables left in the production registry, `export_workspace`
    issues none either. Fourteen statements were captured and not one contained
    `JOIN`. A checker with no input reports no violations, which reads exactly
    like a clean result.
    """

    @pytest.fixture
    def with_indirect(self, database, monkeypatch):
        """Register the indirect fixture, which is what produces a join."""
        from src.workspace import retention
        from src.workspace.retention import WORKSPACE_RECORDS

        from tests import ownership_fixture

        store = WorkspaceStore(POSTGRES_URL)
        ownership_fixture.create(store)
        extended = dict(WORKSPACE_RECORDS)
        extended[ownership_fixture.TABLE] = ownership_fixture.INDIRECT_CHILD
        monkeypatch.setattr(retention, "WORKSPACE_RECORDS", extended)
        return store

    def test_every_read_preserves_composite_identity(self, with_indirect,
                                                     metadata):
        with capture_statements() as issued:
            every_read(with_indirect, A)
        assert any("JOIN" in one.upper() for one in issued), (
            "no join was captured, so this check had nothing to inspect and "
            "would report clean whatever the joins looked like")
        found = consumer_violations(issued, metadata["primary_keys"],
                                    tenant_owned_tables())
        assert found == [], "\n".join(str(one) for one in found)

    def test_deletion_preserves_composite_identity(self, with_indirect,
                                                   metadata):
        every_write(WorkspaceStore(POSTGRES_URL), B)
        with capture_statements() as issued:
            delete_workspace(with_indirect, B,
                             requested_at="2026-07-01T00:00:00Z")
        assert any("JOIN" in one.upper() or " IN (" in one.upper()
                   for one in issued), (
            "the deletion issued no join or subquery to inspect")
        found = consumer_violations(issued, metadata["primary_keys"],
                                    tenant_owned_tables())
        assert found == [], "\n".join(str(one) for one in found)

    def test_the_production_reads_currently_issue_no_ownership_joins(
            self, database, metadata):
        """Recorded, not assumed.

        Every ownership question is answered by an `owner` column now, so there
        is no join left to get wrong on the production read path. That is a
        good state and a fragile one: the first table that becomes indirect
        again brings the whole class of defect back, and this test is where
        that shows up.
        """
        store = WorkspaceStore(POSTGRES_URL)
        with capture_statements() as issued:
            every_read(store, A)
        joins = [one for one in issued if "JOIN" in one.upper()]
        assert joins == [], (
            "an ownership join has appeared on the read path; it must cover "
            f"the whole referenced identity: {joins}")

    def test_it_detects_the_ownership_path_defect(self, metadata):
        """The exact statement the truncated `OwnershipPath` produced."""
        found = consumer_violations(
            ["SELECT child.* FROM child JOIN worksheet_proposal "
             "ON worksheet_proposal.proposal_id = child.proposal_id "
             "WHERE worksheet_proposal.owner = %s"],
            metadata["primary_keys"], tenant_owned_tables())
        assert any("does not include worksheet_proposal.owner" in one.detail
                   for one in found)

    def test_it_accepts_a_join_covering_the_whole_key(self, metadata):
        assert consumer_violations(
            ["SELECT child.* FROM child JOIN worksheet_proposal "
             "ON worksheet_proposal.proposal_id = child.proposal_id "
             "AND worksheet_proposal.owner = child.proposal_owner "
             "WHERE worksheet_proposal.owner = %s"],
            metadata["primary_keys"], tenant_owned_tables()) == []

    def test_it_detects_a_scalar_subquery_against_a_composite_identity(
            self, metadata):
        found = consumer_violations(
            ["DELETE FROM child WHERE proposal_id IN "
             "(SELECT proposal_id FROM worksheet_proposal WHERE owner = %s)"],
            metadata["primary_keys"], tenant_owned_tables())
        assert any("single column" in one.detail for one in found)


class TestTheBehaviourStillHasToAgree:
    """Three static layers can pass while the behaviour still crosses."""

    def test_two_tenants_hold_every_identical_id(self, database):
        every_write(WorkspaceStore(POSTGRES_URL), A)
        every_write(WorkspaceStore(POSTGRES_URL), B)
        conn = Database(POSTGRES_URL).connect()
        try:
            for table, column, value in (
                    ("plan", "plan_id", PLAN),
                    ("plan_run", "run_id", RUN),
                    ("worksheet", "worksheet_id", WORKSHEET),
                    ("planned_event", "planned_event_id", "pe-1"),
                    ("observed_event", "observed_event_id", "oe-1"),
                    ("event_reconciliation", "reconciliation_id", "rc-1")):
                rows = conn.execute(
                    f"SELECT owner FROM {table} WHERE {column} = ? "
                    "ORDER BY owner", (value,)).fetchall()
                owners = [r["owner"] for r in rows]
                assert A in owners and B in owners, f"{table}: {owners}"
        finally:
            conn.close()

    def test_deleting_one_leaves_the_other(self, database):
        every_write(WorkspaceStore(POSTGRES_URL), A)
        every_write(WorkspaceStore(POSTGRES_URL), B)
        delete_workspace(WorkspaceStore(POSTGRES_URL), A,
                         requested_at="2026-07-01T00:00:00Z")
        assert export_workspace(WorkspaceStore(POSTGRES_URL),
                                B)["counts"]["worksheet"] == 1
