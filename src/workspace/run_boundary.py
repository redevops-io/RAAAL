"""The execution boundary: a compiled plan in, a figure and its evidence out.

    runtime artifact ─┐
                      ├─► execute_compiled_plan ─► figure + evidence
    legacy artifact ──┘

**One implementation, two callers.** The pilot path does not get its own copy of
Quantify's execution logic — a second copy would drift, and the drift would show
up as two users getting different numbers from the same plan.

**Legacy execution may be reused; legacy interpretation may not be
reintroduced.** That is the rule this module exists to make structural.
`execute_compiled_plan` takes a `ScenarioSpecification` — something already
compiled — and there is no path through it that turns text into a plan.
`compile_scenario` and `compile_draft` are not imported here and
`test_pilot_route` proves the pilot branch never reaches them.

**Why `stated_text` is still a parameter, and why that is not reinterpretation.**
The coverage gate reads the user's own words to check that every declared
element was actually executed — the gate that caught three prompts returning an
identical $103,393 while each quietly dropped a different declared element. It
*verifies*; it never produces a value. And for a runtime plan it is deliberately
an independent witness: the reader said what the sentence meant, and coverage
asks the raw text whether anything the reader found was then lost on the way to
a figure. Deriving the check from the reader's own output instead would be
asking one witness to confirm itself.

**Why this file is not called `execution.py`.** That name was taken —
`workspace/execution.py` states what an accepted worksheet proposal will
execute — and the first version of this module overwrote it. Nothing warned:
the write reported success, the module imported, and the failure surfaced as a
collection error in a test file whose subject had silently ceased to exist.
Second occurrence of that class in this project.

**The body lives here now.** `_run` and its helpers were defined in `routes.py`
until the move, which was deliberately not bundled with wiring the pilot: a
four-hundred-line move in the same commit as a new execution path would have
made a regression in either indistinguishable from the other. `routes.py`
imports them back for its own call sites, so the dependency runs one way —
routes depend on execution, never the reverse.
"""
from __future__ import annotations

from dataclasses import replace as dataclasses_replace
from typing import Any, Dict, List, Optional

import pandas as pd

from ..mission import (
    CashFlow,
    CashPolicy,
    RunConditions,
    buy_and_hold,
    classify,
    compare,
    comparison_payload,
    hold_cash,
    simulate,
)
from ..mission.rebalance import UnsupportedRebalancing, weighted
from ..mission.scenario import UNSIMULATED
from ..mission.schedule import UnsupportedCadence
from .comparability_record import as_payload as comparability_payload
from .comparability_record import record as comparability_records
from .environment import pins_for


#: Re-exported from the evaluator, which is what decides it now. `routes`
#: imports it from here and continues to, because where a refusal is defined is
#: not the page's business — only what it says.
from ..evaluation.core import STRATEGY_NOT_EXECUTED  # noqa: F401


def declare_unsimulated(scenario, scope: Optional[Dict[str, Any]]
                        ) -> Dict[str, Any]:
    """Add every declared-but-unsimulated behaviour to the modelling scope.

    Derived from the scenario rather than hardcoded, so a behaviour that becomes
    simulatable stops being disclosed by deleting one entry in `UNSIMULATED`,
    and one that is added starts being disclosed without anyone remembering to
    edit this function.
    """
    scope = dict(scope or {})
    # Keyed by what is unsimulated, which is no longer the same as the
    # dimension. `dividend_policy` used to be wholly unmodelled; reinvestment
    # is now run on the snapshot's total-return series and only `held_as_cash`
    # remains. Keying on the dimension would disclose a limitation that no
    # longer applies to the common case — a false disclosure, which is worse
    # than none, because it teaches a reader to discount the true ones.
    declared = {}
    # Only the policy that is still unsimulated. `dividend_policy` used to be
    # disclosed whatever it said, because neither reading was honoured;
    # reinvestment now runs on the total-return series, and disclosing a
    # limitation that no longer applies teaches a reader to discount the ones
    # that do.
    policy = scenario.holdings_policy.dividend_policy
    if policy == "held_as_cash":
        declared["dividend_policy"] = policy
    # The event program was absent from this dict while the docstring above
    # claimed the disclosure was derived rather than hardcoded. It was derived
    # — from a dict with one entry — so a plan whose entire strategy went
    # unexecuted rendered an empty NOT MODELLED column. The claim to be
    # exhaustive is what made the omission dangerous rather than merely
    # incomplete.
    if scenario.event_program:
        declared["event_program"] = f"{len(scenario.event_program)} step(s)"
    not_modelled = {
        field: {"declared": value, "why": UNSIMULATED[field]}
        for field, value in declared.items() if field in UNSIMULATED
    }
    if not_modelled:
        scope["declared_but_not_simulated"] = not_modelled
        # And in the shape the page reads.
        #
        # `declared_but_not_simulated` is the machine-readable record and was
        # the only thing written. `_scope.html` renders `scope.not_modelled`,
        # so this disclosure has never appeared on a page in its life: computed
        # correctly, attached to the result, stored, and displayed by a
        # template reading a different key. Both columns of "What this
        # simulation models" were empty on the plan that prompted this work.
        #
        # Kept as two keys rather than renamed. The stored form is a record
        # older results already carry, and the rendered form is a list a
        # template can iterate; collapsing them would either break the archive
        # or put presentation into the artifact.
        rendered = list(scope.get("not_modelled") or ())
        rendered.extend(
            {"reason": entry["why"], "declared": entry["declared"],
             "field": field}
            for field, entry in sorted(not_modelled.items()))
        scope["not_modelled"] = rendered
    return scope


def execute_compiled_plan(scenario, access, *,
                          scope: Optional[Dict[str, Any]] = None,
                          stated_text: str = "") -> Dict[str, Any]:
    """Simulate a compiled scenario and its benchmarks under one set of flows.

    `access` is the `MarketDataAccess`, never a bare frame: the record of which
    data produced a figure has to travel with the data, because a caller
    attaching provenance afterwards is a caller that can forget — and a run it
    forgot on looks exactly like one it did not.

    Returns the run dict the worksheet renders, including `unavailable` and
    `strategy_not_executed` when the engine refuses. A refusal is a result: the
    figure is absent and the reason is the engine's own.
    """
    return _run(scenario, access, scope, stated_text=stated_text)


def market_data_for(scenario, *, context: str, plan_id: str = ""):
    """The prices this plan needs, with the provenance of where they came from.

    Wrapped for the same reason as above: one accessor, so both paths ask the
    same question and a change to how data is resolved cannot reach one and
    miss the other.
    """
    return _market_data(context, plan_id=plan_id, scenario=scenario)


def _reinvests(scenario) -> bool:
    """Whether this plan's figure should credit distributions.

    Read from the scenario the user confirmed rather than from a constant. The
    two policies are materially different strategies over a long horizon —
    reinvesting compounds the position, holding as cash does not — and the
    engine ran both on price series only, so they produced the same number and
    the choice was recorded without being honoured.
    """
    policy = getattr(getattr(scenario, "holdings_policy", None),
                     "dividend_policy", "")
    return str(policy) == "reinvested"


def _market_data(context: str, *, plan_id: str = "", scenario=None,
                 ran_at: str = ""):
    """The frame, its provenance and the record of this delivery, together.

    The run identity is computed *before* the data is resolved, wherever the
    caller knows what it will store. That is what lets the delivery name the
    execution it was made for, without a second write to connect them later —
    and a chain with a second write has a half-connected state that a crash can
    find. `run_id_for` is deterministic over exactly these three inputs, so
    this is the same identity `generate` will derive, not a guess at it.
    """
    from ..market_data.access import resolve
    from .generate import run_id_for

    run_id = (run_id_for(plan_id, scenario.content_hash, ran_at)
              if plan_id and scenario is not None and ran_at else None)
    return resolve(context=context, run_id=run_id,
                   request_id=f"{context}:{plan_id or 'anonymous'}",
                   reinvested=scenario is not None and _reinvests(scenario))

def _flows_from(schedule, sessions: pd.DatetimeIndex) -> List[CashFlow]:
    """Turn a declared schedule into dated contributions.

    Delegates to `mission.schedule.expand`. Kept as a named function here
    because it is the route's seam and several tests reach for it directly.
    """
    from ..mission.schedule import expand

    return expand(schedule, sessions, cash_flow=CashFlow)

def _benchmark_specs(prices: pd.DataFrame, assets) -> List[Dict[str, Any]]:
    """The set, declared before anything runs and generated by a named rule.

    Cash is always present. It is the comparison nobody asks for and the one that
    answers "was any of this worth doing?".
    """
    universe = [a for a in assets if a in prices.columns]
    specs: List[Dict[str, Any]] = []
    if universe:
        specs.append({"name": "Your basket, bought and held",
                      "tickers": universe, "program": buy_and_hold(universe),
                      "description": "the same instruments, with no timing rule"})
    for ticker, label in (("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"),
                          ("AGG", "Aggregate bonds")):
        specs.append({
            "name": f"Contribute to {label}", "tickers": [ticker],
            "program": buy_and_hold([ticker]),
            "description": f"the same contributions into {ticker}",
        })
    specs.append({"name": "Hold cash", "tickers": [], "program": hold_cash(),
                  "description": "contribute and never invest"})
    return specs

def _resolve_window(scenario, prices):
    """The declared period, resolved against the snapshot's own calendar.

    `None` when the plan declares no period, or declares one this build does
    not support — the compiler has already refused the unsupported kinds, and
    resolving one here would be a second opinion about what it meant.
    """
    from ..mission import time_window
    from ..mission.signals import warmup_sessions

    window = getattr(scenario.provenance, "time_window", None)
    if window is None or not getattr(window, "supported", False):
        return None

    warmup = 0
    if getattr(scenario, "is_event_funded", False):
        warmup = warmup_sessions(scenario.funding.trigger.window)
    return time_window.resolve(
        window, [one.date() for one in prices.index], warmup_sessions=warmup)

def _funding_events(scenario, prices, sessions):
    """Contributions from the scenario's funding policy, or a reason it cannot.

    Returns `(events, signals, unexecutable, error)`. `events is None` means the
    plan is not event-funded and the existing schedule expansion applies —
    which keeps `Scheduled` exactly what it was rather than reimplementing a
    cadence a second time.
    """
    if not scenario.is_event_funded:
        return None, (), (), None

    from ..mission.funding import contribution_events, unexecutable_signals
    from ..mission.signals import UnsupportedSignal

    from ..mission.funding import UnsupportedFunding

    try:
        signals = scenario.funding.trigger.signals(prices)
        events = contribution_events(scenario.funding, frame=prices,
                                     sessions=sessions)
        skipped = unexecutable_signals(scenario.funding, frame=prices,
                                       sessions=sessions)
    except (UnsupportedSignal, UnsupportedFunding) as refused:
        # Refused with its own reason, not the generic one. "There is no price
        # history for SPY" and "this build does not compute exponential
        # averages" send a reader to different places, and the generic message
        # would send them to neither.
        return None, (), (), str(refused)

    if not events:
        # Zero crossings is a legitimate answer and a very misleading figure:
        # the plan contributes nothing, holds nothing, and would report a 0%
        # return that reads as the strategy having performed.
        # Names the period actually evaluated, not "the available history".
        # With the window now applied, a three-month plan legitimately finds no
        # crossing — and telling that user their condition never occurred in
        # "the available history" describes a search they did not ask for.
        window = getattr(scenario.provenance, "time_window", None)
        period = (f"over {window.label}" if getattr(window, "label", "")
                  else "over the period evaluated")
        return None, signals, skipped, (
            f"The condition never occurred {period}, so no purchase was "
            f"triggered and there is no result to report. {len(signals)} "
            f"signal(s) were detected in the sessions examined.")
    return events, signals, skipped, None

def _run(scenario, access, scope: Optional[Dict[str, Any]] = None,
         stated_text: str = "") -> Dict[str, Any]:
    """Evaluate a scenario, then shape and attribute what came back.

    `_run` used to do four jobs its name mentioned none of: the calculation,
    the checks that decide whether a figure may be shown, the benchmarks and
    payload the page renders, and the provenance the result carries. The first
    two are `evaluation.core.evaluate_plan` now. What is left here is the third
    and fourth, which is why they were left: neither changes the figure nor
    whether it is publishable, and moving them into an evaluation service would
    take the application with them.

    Takes the `MarketDataAccess` rather than a bare frame, so the record of
    which data produced the figures travels with the data that produced them.
    """
    from ..evaluation.core import evaluate_plan
    from ..market_data.access import MarketDataAccess

    if not isinstance(access, MarketDataAccess):
        raise TypeError(
            "_run needs the MarketDataAccess the frame came from, not the "
            "frame alone; the provenance is not reconstructable afterwards")

    # PROVENANCE, supplied to the calculation rather than fetched by it.
    #
    # Pins need the sessions this run will actually use, which only exist once
    # the window has been applied — so the evaluator calls back with them
    # instead of this module reaching into the calculation, or the calculation
    # reaching out for the application's pinning machinery.
    pinned: Dict[str, Any] = {}

    def pin_scope(sessions) -> Dict[str, Any]:
        snapshot = f"prices@{sessions[-1].date()}"
        pins = pins_for(scenario, snapshot=snapshot)
        pinned.update({"pins": pins, "snapshot": snapshot,
                       "sessions": sessions})
        return {"unpinned_runtimes": pins.limitations()} if pins.unpinned else {}

    evaluated = evaluate_plan(scenario, access.frame, scope=scope,
                              stated_text=stated_text, pin_scope=pin_scope)

    if not evaluated.publishable:
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None, "strategy_not_executed": True,
                "coverage": evaluated.coverage, "ledger": evaluated.ledger,
                "reconciliation": evaluated.reconciliation,
                # Carried from the refusal site, where the reason is known — so
                # the page can say whether a value fixes it or nothing does.
                "refusal_kind": evaluated.refusal_kind,
                "unavailable": evaluated.refusal}

    result = evaluated.result
    scope = evaluated.scope
    prices = evaluated.prices
    sessions = pinned["sessions"]
    pins = pinned["pins"]
    snapshot = pinned["snapshot"]

    # PROVENANCE. What the engine was actually given, digested at the moment it
    # was given: the delivery digest proves the resolver returned these rows and
    # cannot prove nothing between there and `simulate` changed them. Equal
    # digests mean the transformation was the identity — they differ the moment
    # a window slices, and the difference must name a declared transformation
    # rather than pass unnoticed.
    from ..market_data.access_event import frame_digest

    execution_input = frame_digest(prices)
    if access.access_event is not None:
        scope = {**scope, "execution_input_digest": execution_input,
                 "execution_input_matches_delivery":
                     execution_input == access.access_event.frame_digest}
        result = dataclasses_replace(result, modelling_scope=scope)
    result = dataclasses_replace(result, market_data=access.provenance)
    if evaluated.ledger is not None:
        # On the result, so a stored run carries the evidence for its own
        # figure. A ledger held only in the response would make a saved run
        # unverifiable the moment the page closed.
        scope = {**scope,
                 "rule_events": len(evaluated.ledger.rows),
                 "execution_ledger": evaluated.ledger.to_json(),
                 "reconciliation": evaluated.reconciliation.to_json()}
        result = dataclasses_replace(result, modelling_scope=scope)

    # APPLICATION. Benchmarks, verdicts and the payload the worksheet renders.
    # None of it gates the figure, which is why it stays outside the evaluator.
    specs = _benchmark_specs(prices, list(evaluated.tradeable))
    benchmarks = compare(prices, flows=list(evaluated.flows),
                         cash_policy=CashPolicy.idle(), benchmarks=specs)

    conditions = RunConditions(
        **pins.as_conditions(),
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=CashPolicy.idle().annual_rate,
        tax_treatment=scenario.tax_treatment,
        cost_bps=10.0, execution_lag=1,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=snapshot,
        declared_rule_executed=(True if scenario.event_program else None),
    )
    benchmark_conditions = {
        spec["name"]: RunConditions(
            **{k: v for k, v in conditions.__dict__.items()
               if k != "allocation_rule_hash"},
            allocation_rule_hash=f"benchmark:{spec['name']}")
        for spec in specs
    }
    verdicts = comparability_records(conditions, benchmark_conditions)

    return {
        "result": result,
        "benchmarks": benchmarks,
        "comparability": classify(conditions, conditions),
        "comparability_records": comparability_payload(verdicts),
        "payload": comparison_payload(
            result, benchmarks,
            declared_order=[s["name"] for s in specs],
            rendered_text="",
            user_originated_rule=True,
            platform_generated_action=False,
            portfolio_selection_performed=False,
        ),
        "coverage": evaluated.coverage,
        "ledger": evaluated.ledger,
        "reconciliation": evaluated.reconciliation,
        # Carried out so the page can state the period it reported on. It was
        # computed here, used to slice the frame, and then discarded — so the
        # one screen that shows a figure could not say what span the figure
        # covered, which is the first thing to check when a stated period is
        # not honoured.
        "resolved_window": evaluated.resolved_window,
        "unavailable": None,
    }

