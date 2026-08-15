"""The engine as it runs today, against `evaluate()`, stream by stream.

Step 6. Before the evaluator moves anywhere, the new entry point has to produce
what the old path produces — and "produce what it produces" has to mean more
than a matching final number, because equal headline figures hiding different
wiring is this codebase's most reliable defect.

    application path:  scenario -> execute_compiled_plan -> run dict
    service path:      spec     -> evaluate()            -> EvaluationResult

Both are given the *same* `MarketDataAccess`, not two resolutions of "the same"
data. Two resolutions are two deliveries; comparing across them would measure
the resolver rather than the evaluator, and any difference would be
uninterpretable.

**The seven streams are compared individually**, so a failure names the stage.
A single bundled digest would say the two paths differ and leave somebody
bisecting an engine to find out where.

**Absent is not empty, on both sides.** A scheduled plan has no signal stage;
if one path reported `()` and the other reported "no such stage", a comparison
that flattened them would agree — and would go on agreeing after the extraction
had quietly dropped a stage.
"""
from __future__ import annotations

import os

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.evaluation.service import STREAMS, evaluate
from src.mission.evaluation_policy import declared_policy
from src.mission.from_intent import compile_intent
from src.mission.strategy_spec import from_scenario, to_scenario

ENGINE = "quantify-engine@conformance"
POLICY = declared_policy(data_policy="SYNTHETIC_ONLY", as_of="2026-08-15")

#: One of each shape the engine has a different path for. Not the whole
#: catalogue: this compares two implementations of the same engine, and the
#: cases that matter are the ones that exercise different stages.
CASES = {
    "monthly-contributions": {"assets": "VTI", "amount": "1000",
                              "cadence": "monthly"},
    "annual-contributions": {"assets": "VTI", "amount": "1000",
                             "cadence": "annual"},
    "two-holdings": {"assets": "VTI,BND", "amount": "1000",
                     "cadence": "monthly"},
    "one-off": {"assets": "VTI", "amount": "10000", "cadence": "once"},
    "event-triggered": {"assets": "VOO", "amount": "1000",
                        "observed_assets": "SPY",
                        "trigger_semantics": "crossing_event",
                        "moving_average_window": "200"},
    "never-sells": {"assets": "VTI", "amount": "500", "cadence": "monthly",
                    "sell_action": "never sold any of it"},
}


@pytest.fixture(autouse=True)
def deployment(monkeypatch):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)


def scenario_for(stated):
    canonical = canonicalise(stated)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="conformance",
        utterance_ref="u",
        fields={name: IntentField(value=value, author=Author.MODEL)
                for name, (value, _a) in canonical.fields.items()},
        unresolved=()).seal()
    out = compile_intent(intent, benchmark_rule="a-rule")
    assert out.scenario is not None, [r.detail for r in out.refusals]
    return out.scenario


def both_paths(stated):
    """One delivery, two evaluators.

    The access is resolved once and handed to both. Resolving twice would be
    two deliveries of "the same" data, and a difference between the paths would
    then be unattributable — the resolver and the evaluator would be varying
    together.
    """
    from src.workspace.run_boundary import execute_compiled_plan, market_data_for

    scenario = scenario_for(stated)
    access = market_data_for(scenario, context="conformance")

    application = execute_compiled_plan(scenario, access)
    service = evaluate(from_scenario(scenario), "", evaluation_policy=POLICY,
                       engine_version=ENGINE, access=access,
                       run_plan=execute_compiled_plan)
    return application, service


def application_streams(run, spec):
    """The old path's answer, in the new path's shape.

    Written from the run dict rather than by calling the service helpers, so
    this is genuinely a second reading of the engine's output. Sharing the
    extraction code would make the comparison a tautology — both sides would
    agree because they were the same function.
    """
    from src.evaluation.service import Stream

    result = run.get("result")
    path = getattr(result, "path", None)
    if path is None:
        return {name: Stream(name, False, absent_because="no path")
                for name in STREAMS}

    sessions = tuple(str(one.date()) if hasattr(one, "date") else str(one)
                     for one in path.value.index)
    fills = tuple({"at": str(f.date), "asset": str(f.ticker),
                   "units": str(f.shares), "price": str(f.price),
                   "cash": str(f.notional), "cost": str(f.cost),
                   "reason": str(f.reason)}
                  for f in path.fills)
    orders = tuple([{"at": str(f.date), "asset": str(f.ticker),
                     "notional": str(f.notional), "reason": str(f.reason),
                     "state": "filled"} for f in path.fills]
                   + [{"at": str(o.date), "asset": str(o.ticker),
                       "notional": str(o.notional), "reason": str(o.reason),
                       "state": "unfilled"} for o in path.unfilled])
    contributions = tuple(
        {"at": str(when.date() if hasattr(when, "date") else when),
         "amount": f"{float(value):.10f}"}
        for when, value in path.flows.items() if float(value) > 0.0)
    cash_flows = tuple(
        {"at": str(when.date() if hasattr(when, "date") else when),
         "amount": f"{float(value):.10f}"}
        for when, value in path.flows.items() if float(value) != 0.0)

    signals = (Stream("signals", True,
                      tuple({"at": str(f.date), "asset": str(f.ticker),
                             "fired": True} for f in path.fills))
               if spec.funding.kind == "event_triggered"
               else Stream("signals", False,
                           absent_because="this plan is scheduled, so no "
                                          "signal is evaluated"))
    return {
        "eligible_sessions": Stream("eligible_sessions", True, sessions),
        "signals": signals,
        "contributions": Stream("contributions", True, contributions),
        "orders": Stream("orders", True, orders),
        "fills": Stream("fills", True, fills),
        "cash_flows": Stream("cash_flows", True, cash_flows),
        "metrics": Stream("metrics", True, ()),   # compared separately below
    }


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
class TestTheTwoPathsAgreeStreamForStream:
    def test_each_stream_matches(self, case):
        application, service = both_paths(CASES[case])
        spec = from_scenario(scenario_for(CASES[case]))
        expected = application_streams(application, spec)

        differing = []
        for name in STREAMS:
            if name == "metrics":
                continue                      # its own test, below
            mine, theirs = expected[name], service.streams[name]
            if (mine.produced, mine.rows) != (theirs.produced, theirs.rows):
                differing.append(name)
        assert differing == [], (
            f"{case}: the service path and the application path disagree on "
            f"{differing}. A stream-by-stream comparison exists so this names "
            "the stage rather than the run")

    def test_absent_stays_absent_on_both_sides(self, case):
        """The distinction the comparison would otherwise erase.

        If one path said "no such stage" and the other said "the stage ran and
        produced nothing", a flattened comparison would agree — and would keep
        agreeing after an extraction dropped the stage entirely.
        """
        application, service = both_paths(CASES[case])
        spec = from_scenario(scenario_for(CASES[case]))
        expected = application_streams(application, spec)

        for name in STREAMS:
            if name == "metrics":
                continue
            assert expected[name].produced is service.streams[name].produced, (
                f"{case}/{name}: one path has this stage and the other does not")

    def test_the_figures_agree(self, case):
        """The headline, checked *after* the streams rather than instead.

        A matching final value is necessary and has never been sufficient here:
        a frame digest matched while a plan invested nothing, and two plans
        shared a hash while being priced on different series.
        """
        application, service = both_paths(CASES[case])
        result = application.get("result")
        metrics = {row["metric"]: row for row in service.streams["metrics"].rows}

        assert metrics["time_weighted_final"]["value"] == \
            f"{float(result.time_weighted.iloc[-1]):.10f}"
        assert metrics["periods_per_year"]["value"] == \
            str(result.periods_per_year)
        assert metrics["money_weighted"]["status"] == \
            str(result.money_weighted.status)


class TestTheComparisonWouldNoticeADifference:
    """Without this, every test above passes on two paths that both do nothing.

    The mutation: two genuinely different plans must disagree on the stream
    that carries the difference, and agree on the ones that do not.
    """

    def test_a_different_cadence_shows_up_in_contributions(self):
        _app, monthly = both_paths(CASES["monthly-contributions"])
        _app2, annual = both_paths(CASES["annual-contributions"])

        assert monthly.streams["contributions"].rows \
            != annual.streams["contributions"].rows
        assert monthly.streams["eligible_sessions"].rows \
            == annual.streams["eligible_sessions"].rows, (
            "changing the cadence changed which sessions the plan is eligible "
            "for, so the streams are not independent and a failure could not "
            "name a stage")

    def test_a_different_holding_shows_up_in_fills(self):
        _one, single = both_paths(CASES["monthly-contributions"])
        _two, pair = both_paths(CASES["two-holdings"])

        assets_single = {row["asset"] for row in single.streams["fills"].rows}
        assets_pair = {row["asset"] for row in pair.streams["fills"].rows}
        assert assets_single != assets_pair
