"""The contract, proven on the live path rather than on the type.

    engine outputs -> RSUResultContext -> MissionResult -> WorkspaceStore
                   -> reopened run -> view model

Step 9 defined what a result must carry. This asserts the running system
actually carries it: that the diagnostics survive a real simulation, a real
write, and a real read, and that a run cannot claim to be complete by having
lost the evidence that it is not.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from src.mission.accounting import CashPolicy
from src.mission.rsu_result import (
    CorruptResultContext,
    Presentability,
    RSUResultContext,
    ScopeStatus,
    build,
)
from src.mission.rsu_result import from_json as context_from_json
from src.mission.rsu_result import validate as validate_context
from src.mission.simulate import MissionResult, simulate
from src.runtime.allocation import instruction_for as allocate_for
from src.runtime.allocation import AllocationSchedule, proceeds_from
from src.runtime.concentration import ConcentrationPolicy, assess, solve
from src.runtime.disposition import DispositionSchedule
from src.runtime.disposition import instruction_for as sell_for
from src.runtime.rsu import VestEvent, WithholdingMethod, in_kind_flow_for
from src.workspace.store import NotSaveable, WorkspaceStore

OWNER = "pilot"
SCOPE = {"modelled": ["share delivery"], "out_of_scope": ["capital-gains tax"]}


@pytest.fixture
def sessions():
    return pd.bdate_range("2026-03-02", "2026-04-30")


def vest():
    return VestEvent(grant_id="g1", employer_ticker="ACME",
                     vest_date="2026-03-02", gross_shares=100.0,
                     vest_price_source="p", withholding_rate=0.22,
                     withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
                     market_data_ref="md@1", corporate_action_ref="ca@1")


def messy_run(sessions):
    """One deliberately incomplete run, carrying every diagnostic type."""
    prices = pd.DataFrame({"ACME": 50.0, "VTI": 100.0}, index=sessions)
    arrival, accounting = in_kind_flow_for(vest(), vest_price=50.0)

    # A disposition that never clears its blackout, so it stays unsettled.
    schedule = DispositionSchedule([sell_for(
        vest_ref="g1", grant_ref="g1", asset="ACME", delivered_shares=78.0,
        policy="SELL_ALL_AND_DIVERSIFY",
        delivery_session=pd.Timestamp("2026-03-02"),
        blackouts=[("2026-03-02", "2026-12-31")])])

    result = simulate(prices, flows=[], program=schedule.program(),
                      in_kind=[arrival], cash_policy=CashPolicy.idle(),
                      modelling_scope=dict(SCOPE))
    schedule.reconcile(result.path.fills)

    # Concentration with an unpriced holding, so sizing refuses.
    assessment = assess(holdings={"ACME": 78.0, "VTI": 10.0},
                        prices={"ACME": 50.0}, cash=0.0,
                        employer_asset="ACME",
                        policy=ConcentrationPolicy(target=0.2),
                        measured_at="2026-03-11")
    plan = solve(assessment, price=50.0, held_shares=78.0,
                 policy=ConcentrationPolicy(target=0.2))

    context = build(
        vest_accounting=accounting,
        unpriced_arrivals=result.modelling_scope.get(
            "unpriced_in_kind_arrivals", ()),
        disposition_schedule=schedule,
        concentration_assessment=assessment, concentration_plan=plan,
        realized_concentration=None,
        verdict_rows=(
            {"benchmark_id": "hold", "status": "COMPARABLE",
             "unchecked_dimensions": []},
            {"benchmark_id": "value_matched", "status": "INCOMPARABLE",
             "reason": "cost model differs", "unchecked_dimensions": []},
            {"benchmark_id": "never_ran", "status": "NOT_EVALUATED",
             "reason": "not built", "unchecked_dimensions": []}),
        modelling_scope=SCOPE)

    return MissionResult(
        path=result.path, time_weighted=result.time_weighted,
        money_weighted=result.money_weighted,
        periods_per_year=result.periods_per_year,
        modelling_scope=result.modelling_scope,
        rsu_context=context, requires_rsu_context=True)


def store_with_plan(tmp_path):
    from src.mission.compiler import compile_scenario
    from src.mission.spec import Inference, Provenance
    from src.mission.scenario import ScenarioSpecification

    store = WorkspaceStore(tmp_path / "w.db")
    compiled = compile_scenario(
        "I put $500 into SPY every month in my taxable brokerage and never sell.",
        name="plan-1", version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id="plan-1", owner=OWNER, scenario=scenario,
                    stated_text="seed", saved_at="t0")
    return store


class TestEveryDiagnosticSurvivesTheLivePath:

    def test_engine_stored_and_reopened_contexts_are_identical(
            self, tmp_path, sessions):
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)

        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        reopened = store.rsu_context_of(store.get_run("r1", OWNER))

        assert reopened.to_json() == result.rsu_context.to_json()

    @pytest.mark.parametrize("section,attribute", [
        ("vest_accounting", "cash_remainder"),
        ("disposition", "unsettled_report"),
        ("disposition", "pending_instructions"),
        ("concentration", "missing_prices"),
        ("concentration", "unresolved_inputs"),
        ("comparisons", "verdict_rows"),
    ])
    def test_each_diagnostic_survives_verbatim(self, tmp_path, sessions,
                                               section, attribute):
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        reopened = store.rsu_context_of(store.get_run("r1", OWNER))

        assert getattr(getattr(reopened, section), attribute) == \
            getattr(getattr(result.rsu_context, section), attribute)

    def test_the_unsettled_disposition_is_actually_present(self, sessions):
        """Guards the fixture: a test that survives an empty payload proves
        nothing."""
        context = messy_run(sessions).rsu_context
        assert context.disposition.unsettled_report
        assert context.concentration.missing_prices
        assert len(context.comparisons.verdict_rows) == 3

    def test_incomparable_and_not_evaluated_rows_both_survive(
            self, tmp_path, sessions):
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        reopened = store.rsu_context_of(store.get_run("r1", OWNER))

        statuses = {row["status"] for row in reopened.comparisons.verdict_rows}
        assert {"INCOMPARABLE", "NOT_EVALUATED"} <= statuses


class TestAnIncompleteRunSaysSo:

    def test_the_messy_run_is_not_complete(self, sessions):
        assert messy_run(sessions).rsu_context.presentability \
            is not Presentability.COMPLETE

    def test_a_missing_concentration_price_blocks(self, sessions):
        assert messy_run(sessions).rsu_context.presentability \
            is Presentability.BLOCKED


class TestAnRSURunCannotOmitItsContext:

    def test_declaring_rsu_mechanics_without_a_context_is_refused(
            self, tmp_path, sessions):
        """Stored without one, the diagnostics deciding whether the figure is
        presentable would exist nowhere after the write."""
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        payload = {**result.to_json(), "rsu_context": None}

        with pytest.raises(NotSaveable, match="result context"):
            store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                             result=payload, comparison={})

    def test_a_non_rsu_run_may_omit_it(self, tmp_path, sessions):
        store = store_with_plan(tmp_path)
        prices = pd.DataFrame({"SPY": 100.0}, index=sessions)
        from src.mission.benchmark import buy_and_hold

        plain = simulate(prices, flows=[], program=buy_and_hold([]),
                         cash_policy=CashPolicy.idle(),
                         modelling_scope=dict(SCOPE))
        store.record_run(run_id="r0", plan_id="plan-1", ran_at="t0",
                         result=plain.to_json(), comparison={})
        assert store.rsu_context_of(store.get_run("r0", OWNER)) is None

    def test_the_requirement_is_declared_not_inferred(self, sessions):
        """Inferred from diagnostics, a clean RSU run would be
        indistinguishable from one that never touched RSU mechanics — and that
        is the case where a missing context is least likely to be noticed."""
        assert messy_run(sessions).requires_rsu_context is True


class TestTamperingIsDetected:

    def stored(self, tmp_path, sessions):
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        return store

    def test_an_edited_status_is_caught(self, tmp_path, sessions):
        """`presentability: COMPLETE` beside a failed disposition."""
        store = self.stored(tmp_path, sessions)
        with sqlite3.connect(tmp_path / "w.db") as conn:
            row = conn.execute(
                "SELECT result FROM plan_run WHERE run_id='r1'").fetchone()
            payload = json.loads(row[0])
            payload["rsu_context"]["presentability"] = "COMPLETE"
            conn.execute("UPDATE plan_run SET result=? WHERE run_id='r1'",
                         (json.dumps(payload),))

        with pytest.raises(CorruptResultContext, match="does not match"):
            store.rsu_context_of(store.get_run("r1", OWNER))

    def test_a_removed_section_is_refused(self, tmp_path, sessions):
        """Rendering the remainder would present an edited record as an
        original."""
        store = self.stored(tmp_path, sessions)
        with sqlite3.connect(tmp_path / "w.db") as conn:
            row = conn.execute(
                "SELECT result FROM plan_run WHERE run_id='r1'").fetchone()
            payload = json.loads(row[0])
            del payload["rsu_context"]["disposition"]
            conn.execute("UPDATE plan_run SET result=? WHERE run_id='r1'",
                         (json.dumps(payload),))

        with pytest.raises(CorruptResultContext, match="disposition"):
            store.rsu_context_of(store.get_run("r1", OWNER))

    def test_an_untouched_context_validates(self, tmp_path, sessions):
        store = self.stored(tmp_path, sessions)
        assert store.rsu_context_of(store.get_run("r1", OWNER)) is not None


class TestHistoricalAbsenceIsExplicit:

    def test_a_version_one_record_is_not_declared(self):
        """An older record's silence is evidence that nothing was recorded, not
        that nothing happened."""
        legacy = RSUResultContext()
        assert legacy.scope_status is ScopeStatus.NOT_DECLARED
        assert legacy.presentability is not Presentability.COMPLETE

    def test_the_schema_version_travels(self, sessions):
        assert messy_run(sessions).to_json()["result_schema_version"] == 2

    def test_a_legacy_payload_has_no_context_rather_than_an_empty_one(
            self, tmp_path, sessions):
        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        payload = {k: v for k, v in result.to_json().items()
                   if k not in {"rsu_context", "requires_rsu_context"}}
        payload["result_schema_version"] = 1
        store.record_run(run_id="old", plan_id="plan-1", ran_at="t0",
                         result=payload, comparison={})

        assert store.rsu_context_of(store.get_run("old", OWNER)) is None


class TestTheContextIsCopiedNotRediscovered:

    def test_the_builder_reads_the_objects_that_computed_the_values(self,
                                                                    sessions):
        """A postprocessor re-reading the portfolio would be a second
        interpretation of the execution, and the two would disagree exactly
        where it mattered."""
        import inspect

        from src.mission import rsu_result

        signature = inspect.signature(rsu_result.build)
        for name in ("disposition_schedule", "allocation_execution",
                     "concentration_assessment", "verdict_rows"):
            assert name in signature.parameters

        source = inspect.getsource(rsu_result.build)
        for forbidden in ("simulate(", "holdings[", "recompute"):
            assert forbidden not in source


class TestTelemetryRemainsExpendable:

    def test_deleting_traces_changes_no_result(self, tmp_path, sessions):
        from src.telemetry import TraceStore

        traces = tmp_path / "trace.db"
        TraceStore(traces)

        store = store_with_plan(tmp_path)
        result = messy_run(sessions)
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        before = store.rsu_context_of(store.get_run("r1", OWNER)).to_json()

        traces.unlink()

        after = store.rsu_context_of(store.get_run("r1", OWNER))
        assert after.to_json() == before
        assert after.presentability is Presentability.BLOCKED
