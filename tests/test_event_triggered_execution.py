"""The declared rule executes, and the ledger can prove it did.

The defect this closes: `_run` called `simulate(..., program=buy_and_hold(...))`
whatever the scenario declared, so "buy $1,000 every time SPY crosses below its
200-day average" was replayed as one purchase held to the end. A user caught it
from arithmetic — $1,000 contributed cannot support a rule that fires
repeatedly — because nothing in the system compared what the plan declared with
what the engine did.

So this file asserts the chain end to end:

    price → signal → contribution event → later execution → fill → shares
          → portfolio → reported metrics

and it asserts its own premises first. Several of the mutations below are
survivable on a fixture where the rule happens to do nothing.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.mission.spec import ScenarioAmendment

#: A held instrument this deployment can price, watched by its own crossing.
#: `SPY` is deliberately reserved by the parser when it appears in a signal
#: context — "whenever SPY is below its average" names a condition, not a
#: holding — so a description written with SPY as the holding compiles to no
#: assets at all and would exercise none of this.
DESCRIPTION = ("I buy $1,000 of VOO every time the S&P 500 crosses below its "
               "200-day moving average for the past 5 years.")

SETTLED = (ScenarioAmendment(question_id="trigger_semantics",
                             answer="crossing_event",
                             recorded_at="2026-08-05T00:00:00Z"),
           # Settled so the account dimension is pinned. Left open, the
           # comparability classifier refuses `attribution_isolated` for an
           # unchecked dimension — correctly, and for a reason that has nothing
           # to do with whether the rule executed.
           ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                             recorded_at="2026-08-05T00:00:00Z"))


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def deployment():
    from src.deploy.context import bind, resolve, unbind

    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


@pytest.fixture
def executed(deployment):
    """The compiled plan, the access it ran against, and the run."""
    from src.mission.compiler import compile_scenario

    import src.workspace.routes as routes

    access = routes._market_data("test")
    plan = compile_scenario(
        DESCRIPTION, name="p", version=1, amendments=SETTLED,
        benchmark_rule="benchmark-policy/public-default@1",
        priceable=tuple(access.frame.columns))
    return plan, access, routes._run(plan.scenario, access)


class TestThePremises:
    """Every mutation below is survivable on a fixture where nothing happens."""

    def test_the_condition_actually_occurs(self, executed):
        _, _, run = executed
        assert run["ledger"].signals, (
            "no crossing in this fixture; every execution assertion would hold "
            "against an engine that executes nothing")

    def test_purchases_were_actually_filled(self, executed):
        _, _, run = executed
        assert run["ledger"].filled_shares > 0

    def test_the_rule_changes_the_answer(self, deployment):
        """The premise that matters, and not the one it first appeared to be.

        "The strategy differs from buy-and-hold" is **false here, correctly**:
        once the rule governs *funding*, the flow-matched buy-and-hold of the
        same basket receives the same contributions and buys the same
        instrument, so the two are identical by construction. Asserting a
        difference there would have failed against correct code.

        The rule's effect is on *when money arrives*, so the counterfactual is
        a different funding policy — the same total, contributed monthly.
        """
        from src.mission.accounting import CashFlow, CashPolicy
        from src.mission.benchmark import buy_and_hold
        from src.mission.funding import EventTriggered, Trigger, contribution_events
        from src.mission.simulate import simulate

        import src.workspace.routes as routes

        frame = routes._market_data("test").frame
        policy = EventTriggered(trigger=Trigger(subject="VOO", window=200),
                                amount=Decimal("1000"))
        events = contribution_events(policy, frame=frame)
        assert events

        triggered = [CashFlow(date=e.session, amount=float(e.amount))
                     for e in events]
        total = float(sum(e.amount for e in events))
        months = pd.date_range(frame.index[0], frame.index[-1], freq="MS")
        landings = [frame.index[frame.index.searchsorted(m)] for m in months
                    if frame.index.searchsorted(m) < len(frame.index)]
        scheduled = [CashFlow(date=s, amount=total / len(landings))
                     for s in landings]

        def final(flows):
            return simulate(frame, flows=flows, program=buy_and_hold(["VOO"]),
                            cash_policy=CashPolicy.idle()).final_value

        assert abs(final(triggered) - final(scheduled)) > 1.0, (
            "event-triggered funding and a flat schedule of the same total "
            "produce the same answer on this fixture, so nothing here can "
            "detect the funding policy being ignored")


class TestTheChainExecutes:
    def test_a_figure_is_produced(self, executed):
        _, _, run = executed
        assert run["unavailable"] is None
        assert run["result"] is not None

    def test_one_purchase_per_executable_signal(self, executed):
        _, _, run = executed
        ledger = run["ledger"]
        assert len(ledger.rows) == len(ledger.signals) - len(ledger.unexecutable)

    def test_more_than_one_purchase(self, executed):
        """The arithmetic the user noticed. One purchase against a repeating
        rule is the shape of the original defect."""
        _, _, run = executed
        assert len(run["ledger"].rows) > 1

    def test_the_total_is_the_amount_times_the_purchases(self, executed):
        _, _, run = executed
        ledger = run["ledger"]
        assert ledger.total_contributed == Decimal("1000") * len(ledger.rows)

    def test_every_purchase_executes_after_its_signal(self, executed):
        """Trading the close that produced the signal reads one bar into the
        future — the most common way a backtest flatters itself."""
        _, _, run = executed
        # Against the *contribution* date. Checked against the execution date
        # alone this passed with the engine funding on the signal's own
        # session, because the fill lags the order by one — the look-ahead sat
        # between the first two dates and the check only looked at the last.
        assert all(row.signal_session < row.contribution_session
                   for row in run["ledger"].rows)
        assert all(row.contribution_session <= row.execution_session
                   for row in run["ledger"].rows)

    def test_every_purchase_bought_shares_at_a_price(self, executed):
        """Both were zero when the ledger joined fills by contribution date,
        and the reconciliation said the run agreed."""
        _, _, run = executed
        assert all(row.shares > 0 and row.price > 0
                   for row in run["ledger"].rows)

    def test_no_fill_is_used_twice(self, executed):
        """Share totals could match while two rows claimed one purchase."""
        _, _, run = executed
        ledger = run["ledger"]
        assert ledger.total_shares == ledger.filled_shares

    def test_the_money_closes(self, executed):
        """Purchases + fees + cash still held == everything contributed.

        Independent of returns and of every benchmark, and it catches the class
        the counting checks cannot: each purchase exists, each fill exists, the
        share totals agree, and money quietly appears or disappears between
        them.
        """
        _, _, run = executed
        ledger = run["ledger"]
        assert run["reconciliation"].checks["the_money_closes"], (
            run["reconciliation"].detail["the_money_closes"])
        # And the premise: there is money and there are fees to account for,
        # or the identity holds trivially at zero.
        assert ledger.filled_notional > 0
        assert ledger.filled_fees > 0

    def test_the_reconciliation_agrees(self, executed):
        _, _, run = executed
        assert run["reconciliation"].agrees, run["reconciliation"].failures()

    def test_the_warm_up_is_honoured(self, executed):
        """A 200-session average is undefined until 200 sessions exist. Emitted
        from a partial window, the first crossing lands in the first weeks of
        every plan, where a user is least able to judge it."""
        _, access, run = executed
        first = run["ledger"].signals[0].session
        # `window - 1`, not `window`. Two hundred observations occupy indices
        # 0..199, so the average is first defined on index 199 — which is the
        # 200th session. Asserting `>= 200` demanded that the first legitimate
        # session be discarded.
        assert access.frame.index.get_loc(first) >= 200 - 1


class TestTheSignalIsTheRuleTheUserChose:
    def test_a_crossing_fires_once_per_drawdown(self, deployment):
        from src.mission.signals import SignalKind, moving_average_signals

        import src.workspace.routes as routes

        frame = routes._market_data("test").frame
        crossings = moving_average_signals(frame, subject="VOO", window=200)
        below = moving_average_signals(
            frame, subject="VOO", window=200,
            kind=SignalKind.BELOW_MOVING_AVERAGE)
        assert 0 < len(crossings) < len(below), (
            "every day below the average is being counted as a crossing")

    def test_the_window_changes_the_signals(self, deployment):
        """50-day and 200-day are different rules. A default would answer a
        question the user did not ask."""
        from src.mission.signals import moving_average_signals

        import src.workspace.routes as routes

        frame = routes._market_data("test").frame
        assert (moving_average_signals(frame, subject="VOO", window=50)
                != moving_average_signals(frame, subject="VOO", window=200))

    def test_an_unstated_window_is_unresolved_not_defaulted(self, deployment):
        from src.mission.compiler import compile_scenario

        plan = compile_scenario(
            "I buy $1,000 of VOO every time it crosses below its moving average.",
            name="p", version=1, amendments=SETTLED, benchmark_rule="b")
        assert any(one.field == "moving_average_window"
                   for one in plan.scenario.provenance.unresolved)

    def test_an_unsupported_estimator_is_refused(self, deployment):
        from src.mission.signals import (
            Estimator,
            UnsupportedSignal,
            moving_average_signals,
        )

        import src.workspace.routes as routes

        frame = routes._market_data("test").frame
        with pytest.raises(UnsupportedSignal):
            moving_average_signals(frame, subject="VOO", window=200,
                                   estimator=Estimator.EXPONENTIAL)

    def test_the_first_and_last_valid_crossings_survive(self, deployment):
        """Off-by-one at either end silently changes the plan. Checked against
        an independently computed mask rather than against the function's own
        output."""
        from src.mission.signals import moving_average_signals

        import src.workspace.routes as routes

        frame = routes._market_data("test").frame
        closes = frame["VOO"].astype(float)
        average = closes.rolling(200, min_periods=200).mean()
        below = closes < average
        expected = frame.index[
            (below & ~below.shift(1).fillna(False).astype(bool)
             & average.notna()).to_numpy()]
        produced = [s.session for s in
                    moving_average_signals(frame, subject="VOO", window=200)]
        assert produced[0] == expected[0]
        assert produced[-1] == expected[-1]
        assert len(produced) == len(expected)


class TestFundingIsOneConceptWithTwoPolicies:
    def test_an_event_funded_plan_states_no_cadence(self, executed):
        plan, _, _ = executed
        assert plan.scenario.is_event_funded
        assert plan.scenario.flow_schedule.cadence == "event_triggered"
        assert plan.scenario.flow_schedule.amount == 0

    def test_stating_both_is_a_conflict(self, executed):
        """Structurally hard, and refused on the compiled form as well —
        a check that only reads the compiler trusts the compiler."""
        import dataclasses

        plan, _, _ = executed
        contradictory = dataclasses.replace(
            plan.scenario,
            flow_schedule=dataclasses.replace(
                plan.scenario.flow_schedule, cadence="monthly", amount=500.0))
        assert any("event-triggered" in conflict
                   for conflict in contradictory.self_conflicts())

    def test_cadence_is_never_asked_for_an_event_funded_plan(self, executed):
        """Not asked, rather than asked and discarded. The user answered
        "once" and it became one contribution for a five-year rule."""
        plan, _, _ = executed
        assert not any(one.field == "cadence"
                       for one in plan.scenario.provenance.unresolved)

    def test_a_scheduled_plan_still_asks_it(self, deployment):
        """The suppression must discriminate, or it is just a deleted
        question."""
        from src.mission.compiler import compile_scenario

        plan = compile_scenario("I buy $500 of VOO.", name="p", version=1,
                                benchmark_rule="b")
        assert any(one.field == "cadence"
                   for one in plan.scenario.provenance.unresolved)

    def test_the_series_the_condition_uses_is_disclosed(self, executed):
        """An index is not priceable here, so the condition is evaluated on the
        instrument held. That is a real assumption with a real consequence and
        the user is entitled to reject it."""
        plan, _, _ = executed
        signal_series = [one for one in plan.scenario.provenance.inferred
                         if one.field == "signal_series"]
        assert signal_series
        assert signal_series[0].value == "VOO"
        assert not signal_series[0].confirmed


class TestBenchmarksShareTheContributions:
    def test_every_benchmark_receives_the_same_total(self, executed):
        _, _, run = executed
        totals = {round(b.result.path.contributed, 2)
                  for b in run["benchmarks"]}
        assert totals == {round(run["result"].path.contributed, 2)}

    def test_the_total_is_the_triggered_one(self, executed):
        """Not the single $1,000 the old schedule produced. A benchmark
        contributed once and compared against a rule contributing thirty times
        would flatter whichever received more money."""
        _, _, run = executed
        assert run["result"].path.contributed == float(
            run["ledger"].total_contributed)

    def test_the_benchmarks_differ_from_each_other(self, executed):
        """Otherwise "the only difference is what the money bought" is true and
        empty."""
        _, _, run = executed
        finals = {round(b.result.final_value, 2) for b in run["benchmarks"]}
        assert len(finals) > 1

    def test_the_strategy_effect_claim_is_now_permitted(self, executed):
        """Refused in Deployment 1 because no rule had run. Granted only once
        a ledger reconciled against the result — the claim rests on the
        comparison having passed, not on the code believing it ran."""
        _, _, run = executed
        verdict = run["comparability"]
        assert "not executed" not in verdict.detail, (
            "the claim is still refused because of the rule")
        assert verdict.attribution_isolated, verdict.detail


class TestTheWarmUpOnConstructedData:
    """The pilot fixture cannot falsify this one.

    Removing `min_periods` produced the identical 30 signals with the first
    still at index 199 — the synthetic series simply contains no crossing that
    a partial-window average would have fired early. The mutation survived, and
    reading the test would never have shown why.

    So the discriminating input is constructed here rather than hoped for: a
    series that falls from the first session, where a one-session average
    crosses immediately and a 200-session average cannot exist yet.
    """

    @staticmethod
    def falling(sessions: int = 260):
        import numpy as np

        index = pd.bdate_range("2020-01-01", periods=sessions)
        # Down every day. A partial-window average sits above the close from
        # session two onward, so an unwarmed indicator fires almost at once.
        closes = np.linspace(100.0, 60.0, sessions)
        return pd.DataFrame({"VOO": closes}, index=index)

    def test_no_signal_before_the_window_is_full(self):
        from src.mission.signals import moving_average_signals

        frame = self.falling()
        signals = moving_average_signals(frame, subject="VOO", window=200)
        assert signals, "this series produces no crossing at all"
        first = frame.index.get_loc(signals[0].session)
        assert first >= 199, (
            f"a signal fired at index {first}, before 200 sessions existed; "
            f"the average it rests on was computed from a partial window")

    def test_an_unwarmed_average_would_have_fired_early(self):
        """The premise. Without it the assertion above passes on any series
        with no early crossing — which is exactly how the pilot fixture hid
        this."""
        import numpy as np

        frame = self.falling()
        closes = frame["VOO"].astype(float)
        unwarmed = closes.rolling(window=200, min_periods=1).mean()
        below = closes < unwarmed
        crossings = below & ~below.shift(1).fillna(False).astype(bool)
        early = [i for i, fired in enumerate(crossings.to_numpy()) if fired
                 and i < 199]
        assert early, (
            "even an unwarmed average produces no early crossing here, so the "
            "case above cannot discriminate")


class TestTheTimelineIsRenderedFromTheLedger:
    """The chart is a view of the evidence, not a second calculation.

    Every number on the page comes from `run.ledger`. A template that
    recomputed a total would be a second answer to the same question, and the
    one nobody checks is the one that drifts — which is how a page came to show
    a figure for a rule that never ran.
    """

    @pytest.fixture
    def page(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()

        access = routes._market_data("test")
        plan = compile_scenario(
            DESCRIPTION, name="p", version=1, amendments=SETTLED,
            benchmark_rule="benchmark-policy/public-default@1",
            priceable=tuple(access.frame.columns))
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id="plan-x", owner="pilot", scenario=scenario,
                        stated_text=DESCRIPTION, saved_at="2026-08-05T00:00:00Z",
                        title="200DMA")
        client = TestClient(api.app)
        response = client.get("/workspace/plans/plan-x")
        assert response.status_code == 200, response.text
        return response.text, routes._run(scenario, access)

    def test_the_summary_states_the_four_numbers(self, page):
        body, run = page
        summary = run["ledger"].summary()
        assert "Signals detected" in body
        assert str(summary["signals_detected"]) in body
        assert "Purchases executed" in body

    def test_every_stored_purchase_appears(self, page):
        """Omitting one row would leave a total nothing on the page supports.
        Checked per execution date rather than by counting `<tr>`, which would
        pass on a table of the right length and the wrong contents."""
        body, run = page
        missing = [str(row.execution_session.date())
                   for row in run["ledger"].rows
                   if str(row.execution_session.date()) not in body]
        assert not missing, f"{len(missing)} purchases are absent: {missing[:3]}"

    def test_every_signal_date_appears(self, page):
        body, run = page
        missing = [str(row.signal_session.date())
                   for row in run["ledger"].rows
                   if str(row.signal_session.date()) not in body]
        assert not missing

    def test_the_three_dates_are_distinct_columns(self, page):
        """Signal, contribution and execution. Collapsing the middle one is
        what let a policy that funded on its own signal session pass the
        look-ahead check."""
        body, _ = page
        assert "Signal date" in body
        assert "Contribution date" in body
        assert "Execution date" in body

    def test_the_cumulative_total_reaches_the_ledger_total(self, page):
        body, run = page
        assert "{:,.0f}".format(run["ledger"].total_contributed) in body

    def test_the_chart_marks_one_point_per_purchase(self, page):
        """A view of the rows, not its own crossing detection."""
        import src.workspace.routes as routes

        _, run = page
        access = routes._market_data("test")
        chart = routes._timeline_chart(
            run["ledger"].rows and _scenario_of(run) or None,
            access.frame, run["ledger"])
        assert chart is None or len(chart["marks"]) == len(run["ledger"].rows)


def _scenario_of(run):
    """The scenario a run was produced from, for the chart check."""
    from src.mission.compiler import compile_scenario

    import src.workspace.routes as routes

    access = routes._market_data("test")
    return compile_scenario(
        DESCRIPTION, name="p", version=1, amendments=SETTLED,
        benchmark_rule="benchmark-policy/public-default@1",
        priceable=tuple(access.frame.columns)).scenario


class TestAFailingReconciliationYieldsNoFigure:
    """Where the strategy-effect claim is actually enforced.

    `declared_rule_executed` is `True` for any event-funded run that reaches
    the classifier — because a run whose ledger disagrees with its result never
    gets there. That early return is the control, and this is the test of it: a
    guard restated in two places is a guard that can drift, and the copy nobody
    reaches is the one that rots.
    """

    def test_a_disagreeing_ledger_suppresses_the_result(self, deployment,
                                                        monkeypatch):
        from src.mission import ledger as ledger_module
        from src.mission.compiler import compile_scenario

        import src.workspace.routes as routes

        access = routes._market_data("test")
        plan = compile_scenario(
            DESCRIPTION, name="p", version=1, amendments=SETTLED,
            benchmark_rule="benchmark-policy/public-default@1",
            priceable=tuple(access.frame.columns))

        # A reconciliation that refuses, without breaking anything upstream —
        # so the failure under test is the disagreement itself and not a
        # missing ledger or an exception on the way to it.
        real = ledger_module.reconcile

        def disagrees(ledger, result):
            verdict = real(ledger, result)
            return ledger_module.Reconciliation(
                agrees=False,
                checks={**verdict.checks, "shares_match_what_the_engine_filled": False},
                detail={**verdict.detail,
                        "shares_match_what_the_engine_filled": "forced"})

        monkeypatch.setattr(ledger_module, "reconcile", disagrees)
        run = routes._run(plan.scenario, access)

        assert run["result"] is None
        assert run["benchmarks"] == []
        assert run["comparability"] is None
        assert "do not agree" in run["unavailable"]

    def test_without_the_forced_failure_it_produces_one(self, executed):
        """The premise. Otherwise the case above passes against a build that
        never produces a figure at all."""
        _, _, run = executed
        assert run["result"] is not None
