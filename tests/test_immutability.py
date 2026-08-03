"""Immutable bodies, checked against the statements PostgreSQL receives.

Several store methods promise immutability in prose — "immutable from here",
"revisions are never edited", "a plan revisited next year must show the result
it actually got". This file is what makes those promises checkable, and the
first time it ran it found five statements doing the opposite of what their
docstring said.

**Captured after dialect translation.** The statement the store writes is not
the one the database receives: `INSERT OR REPLACE` becomes
`ON CONFLICT ... DO UPDATE SET payload = EXCLUDED.payload`, and it is the
rewrite that overwrites a body. Checking the source would have found the words
"immutable" and stopped there — the ninth prose-matching failure in this
codebase was exactly that, so the rule here is captured statements only.

Two layers, deliberately:

    captured statements   the dangerous SQL cannot be sent
    behaviour             redelivery is accepted, divergence is a conflict,
                          and the stored body is unchanged either way

Either alone is insufficient. The static check cannot see what the API means;
the behavioural check cannot see a statement that is dangerous but not yet
exercised by a test.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import os

import pytest

from src.db.engine import Database, capture_statements
from src.db.mutability import (
    TABLE_MUTABILITY,
    TableClass,
    inspect_statement,
    violations,
)
from src.mission.rsu_reconcile import (
    EventReconciliation,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
)
from src.workspace.store import NotSaveable, WorkspaceStore
from src.workspace.worksheet import create, revise, from_json

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

OWNER = "alice"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}


def scenario_for(name="p1"):
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


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def exercise_every_write(store):
    """One call to each writing method, under capture."""
    store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                    stated_text="x", saved_at="2026-01-01T00:00:00Z")
    store.record_run(run_id="r1", plan_id="p1", ran_at="2026-01-01T00:00:00Z",
                     result=RESULT, comparison={}, owner=OWNER)
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="p1", primary_run_ref="r1",
                                created_at="2026-01-01T00:00:00Z"))
    store.record_planned_event(
        owner=OWNER, worksheet_id="ws-1",
        event=PlannedEvent(event_id="pe1", grant_ref="g",
                           expected_date="2026-06-15"),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1")
    store.record_observed_event(
        owner=OWNER, worksheet_id="ws-1",
        event=ObservedEvent(observation_id="oe1", observed_date="2026-06-16",
                            effective_date="2026-06-15"),
        created_at="2026-01-01T00:00:00Z")
    store.record_reconciliation(
        owner=OWNER, worksheet_id="ws-1",
        reconciliation=EventReconciliation(
            reconciliation_id="rc1", status=ReconciliationStatus.PENDING,
            derived_at="2026-06-17T00:00:00Z"))
    store.record_confirmation_event(
        event_id="ce1", owner=OWNER, occurred_at="2026-01-01T00:00:00Z",
        kind="EDIT", path=None, field=None, provenance=None,
        original_value=None, final_value=None, reason=None,
        compiler_version=None, defaults_ref=None)


class TestTheClassificationIsExhaustive:
    def test_every_table_is_classified(self):
        """A new table must fail here rather than default to mutable."""
        from src.db.schema import metadata

        assert set(TABLE_MUTABILITY) == set(metadata.tables), (
            "unclassified: "
            f"{set(metadata.tables) - set(TABLE_MUTABILITY)}")

    def test_every_immutable_table_names_its_protected_columns(self):
        for policy in TABLE_MUTABILITY.values():
            if policy.kind is TableClass.MUTABLE_PROJECTION:
                continue
            assert policy.immutable_columns, (
                f"{policy.table} is {policy.kind.value} and protects no "
                "columns, so the classification enforces nothing")

    def test_every_classification_records_why(self):
        for policy in TABLE_MUTABILITY.values():
            assert policy.rationale.strip(), policy.table


class TestNoStoreWriteTouchesAnImmutableBody:
    """The check that found five violations the first time it ran."""

    def test_a_full_pass_over_the_store_issues_nothing_forbidden(self, store):
        with capture_statements() as issued:
            exercise_every_write(store)
        assert issued, "nothing was captured — the recorder is not wired in"
        found = violations(issued)
        assert found == [], "\n".join(str(one) for one in found)

    @pytest.mark.skipif(not POSTGRES_URL, reason="needs PostgreSQL")
    def test_the_same_holds_for_the_statements_postgresql_receives(self):
        """The translated statement is the one that matters."""
        from sqlalchemy import text

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        database.create_all()

        with capture_statements() as issued:
            exercise_every_write(WorkspaceStore(POSTGRES_URL))
        assert any("ON CONFLICT" in one for one in issued), (
            "no upsert was captured; this test is not seeing translated SQL")
        found = violations(issued)
        assert found == [], "\n".join(str(one) for one in found)

    def test_the_worksheet_revision_write_is_a_plain_insert(self, store):
        with capture_statements() as issued:
            store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                            stated_text="x", saved_at="2026-01-01T00:00:00Z")
            store.save_worksheet(create(
                worksheet_id="ws-1", owner_id=OWNER, scenario_ref="p1",
                primary_run_ref="r1", created_at="2026-01-01T00:00:00Z"))
        inserts = [one for one in issued
                   if "INSERT" in one.upper() and "worksheet" in one]
        assert inserts
        assert not any("OR REPLACE" in one.upper() or "DO UPDATE" in one.upper()
                       for one in inserts)


class TestTheGuardDetectsWhatItShould:
    """Constructed statements, so the detector is exercised directly."""

    def test_an_upsert_overwriting_a_body_is_a_violation(self):
        found = inspect_statement(
            "INSERT INTO plan_run (owner, run_id, result) VALUES (%s,%s,%s) "
            "ON CONFLICT (owner, run_id) DO UPDATE SET result = EXCLUDED.result")
        assert found and "result" in found[0].reason

    def test_an_update_assigning_a_body_column_is_a_violation(self):
        found = inspect_statement(
            "UPDATE planned_event SET payload = %s WHERE owner = %s")
        assert found and "payload" in found[0].reason

    def test_jsonb_set_on_an_immutable_body_is_a_violation(self):
        found = inspect_statement(
            "UPDATE observed_event SET payload = "
            "jsonb_set(payload, '{a}', '1') WHERE owner = %s")
        assert found
        assert any("jsonb_set" in one.reason for one in found)

    def test_concatenation_is_a_violation(self):
        found = inspect_statement(
            "UPDATE planned_event SET payload = payload || '{\"a\":1}' "
            "WHERE owner = %s")
        assert found

    def test_path_removal_is_a_violation(self):
        found = inspect_statement(
            "UPDATE planned_event SET payload = payload #- '{a}' "
            "WHERE owner = %s")
        assert found

    def test_a_lifecycle_status_update_is_allowed(self):
        """Not everything that updates is forbidden — a proposal's outcome has
        to move, and only its body is protected."""
        assert inspect_statement(
            "UPDATE worksheet_proposal SET status = %s, resolved_at = %s "
            "WHERE proposal_id = %s AND owner = %s AND status = 'PROPOSED'") == []

    def test_a_projection_may_be_replaced(self):
        """A reconciliation is re-derivable, so replacing it loses nothing."""
        assert inspect_statement(
            "INSERT INTO event_reconciliation (owner, payload) VALUES (%s,%s) "
            "ON CONFLICT (owner) DO UPDATE SET payload = EXCLUDED.payload") == []

    def test_do_nothing_is_allowed(self):
        assert inspect_statement(
            "INSERT INTO plan_run (owner, run_id) VALUES (%s,%s) "
            "ON CONFLICT (owner, run_id) DO NOTHING") == []


class TestRedeliveryAndDivergence:
    """The API semantics the captured check cannot see."""

    def test_an_identical_plan_is_accepted_as_redelivery(self, store):
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        assert store.save_plan(plan_id="p1", owner=OWNER,
                               scenario=scenario_for(), stated_text="x",
                               saved_at="2026-01-02T00:00:00Z") == "p1"

    def test_a_divergent_plan_is_a_conflict(self, store):
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        with pytest.raises(NotSaveable, match="different contents"):
            store.save_plan(plan_id="p1", owner=OWNER,
                            scenario=scenario_for(name="something-else"),
                            stated_text="x", saved_at="2026-01-02T00:00:00Z")

    def test_the_original_plan_body_is_unchanged(self, store):
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        before = store.get_plan("p1", OWNER)["content_hash"]
        with pytest.raises(NotSaveable):
            store.save_plan(plan_id="p1", owner=OWNER,
                            scenario=scenario_for(name="something-else"),
                            stated_text="x", saved_at="2026-01-02T00:00:00Z")
        assert store.get_plan("p1", OWNER)["content_hash"] == before

    def test_an_identical_run_is_accepted_as_redelivery(self, store):
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        store.record_run(run_id="r1", plan_id="p1",
                         ran_at="2026-01-01T00:00:00Z", result=RESULT,
                         comparison={}, owner=OWNER)
        assert store.record_run(run_id="r1", plan_id="p1",
                                ran_at="2026-01-02T00:00:00Z", result=RESULT,
                                comparison={}, owner=OWNER) == "r1"

    def test_a_divergent_run_is_a_conflict(self, store):
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        store.record_run(run_id="r1", plan_id="p1",
                         ran_at="2026-01-01T00:00:00Z", result=RESULT,
                         comparison={}, owner=OWNER)
        with pytest.raises(NotSaveable, match="different result"):
            store.record_run(run_id="r1", plan_id="p1",
                             ran_at="2026-01-02T00:00:00Z",
                             result={**RESULT, "final_value": 2.0},
                             comparison={}, owner=OWNER)

    def test_the_original_run_verdict_is_unchanged(self, store):
        """The reason the whole thing matters: a saved worksheet cites this
        run, and must keep showing the figure it cited."""
        store.save_plan(plan_id="p1", owner=OWNER, scenario=scenario_for(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        store.record_run(run_id="r1", plan_id="p1",
                         ran_at="2026-01-01T00:00:00Z", result=RESULT,
                         comparison={}, owner=OWNER)
        with pytest.raises(NotSaveable):
            store.record_run(run_id="r1", plan_id="p1",
                             ran_at="2026-01-02T00:00:00Z",
                             result={**RESULT, "final_value": 2.0},
                             comparison={}, owner=OWNER)
        assert store.get_run("r1", OWNER)["result"]["final_value"] == 1.0
