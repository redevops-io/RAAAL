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


STRATEGY_NOT_EXECUTED = (
    "This result is unavailable. The plan contains a conditional purchase "
    "rule, but this version of Quantify did not execute that rule. No strategy "
    "result is shown. Your description and clarifications remain saved."
)


def declare_unsimulated(scenario, scope: Optional[Dict[str, Any]]
                        ) -> Dict[str, Any]:
    """Add every declared-but-unsimulated behaviour to the modelling scope.

    Derived from the scenario rather than hardcoded, so a behaviour that becomes
    simulatable stops being disclosed by deleting one entry in `UNSIMULATED`,
    and one that is added starts being disclosed without anyone remembering to
    edit this function.
    """
    scope = dict(scope or {})
    declared = {
        "dividend_policy": scenario.holdings_policy.dividend_policy,
    }
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
                   request_id=f"{context}:{plan_id or 'anonymous'}")

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
    """Simulate a scenario and its benchmarks under identical conditions.

    Takes the `MarketDataAccess` rather than a bare frame, so the record of
    which data produced the figures travels with the data that produced them.
    A caller attaching provenance to the result afterwards is a caller that can
    forget, and the run it forgot on looks exactly like one it did not.
    """
    from ..market_data.access import MarketDataAccess
    from ..mission import coverage as coverage_module

    if not isinstance(access, MarketDataAccess):
        raise TypeError(
            "_run needs the MarketDataAccess the frame came from, not the "
            "frame alone; the provenance is not reconstructable afterwards")
    prices = access.frame

    # The evaluation period the user asked for, applied.
    #
    # `time_window.detect` has always run and always refused the kinds this
    # build cannot replay. `time_window.resolve` — which turns a supported
    # instruction into start, end and warm-up dates — had no callers at all,
    # so "over the past 3 months" replayed 2,488 sessions of history and
    # returned $25,000 contributed. Sixth instance of a mechanism that was
    # present, correct and unreached.
    #
    # The warm-up is kept *outside* the reported period and inside the frame:
    # a 200-session average needs 200 sessions before the first day of the
    # window, or the earliest crossings cannot be evaluated at all.
    resolved_window = _resolve_window(scenario, prices)
    if resolved_window is not None:
        first = resolved_window.warmup_start or resolved_window.start
        prices = prices.loc[str(first):str(resolved_window.end)]

    sessions = prices.index
    policy = CashPolicy.idle()
    assets = list(scenario.allocation_rule.assets)
    tradeable = [a for a in assets if a in prices.columns]

    # One funding policy, one set of dated contributions, and the benchmarks
    # receive exactly those. That is what makes "every dimension outside the
    # rule was held identical" true rather than aspirational: the schedule is
    # not merely equivalent between plan and benchmark, it is the same object.
    events, ledger_signals, unexecutable, funding_error = _funding_events(
        scenario, prices, sessions)
    if funding_error is not None:
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None, "strategy_not_executed": True,
                "coverage": None, "unavailable": funding_error}
    try:
        flows = (tuple(CashFlow(date=event.session, amount=float(event.amount))
                       for event in events)
                 if events is not None
                 else _flows_from(scenario.flow_schedule, sessions))
    except UnsupportedCadence as refused:
        # The same shape as a refused funding policy above: no figure, and the
        # reason the engine gave rather than a generic one.
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None, "strategy_not_executed": True,
                "coverage": None, "unavailable": str(refused)}

    # A declared conditional rule that this engine does not execute produces no
    # figure at all.
    #
    # `simulate` is called with `program=buy_and_hold(tradeable)` regardless of
    # what the scenario declares, and nothing converts `event_program` into an
    # `EventProgram`. A plan reading "buy $1,000 every time SPY crosses below
    # its 200-day average" was therefore replayed as a single purchase held to
    # the end — and returned a figure identical to the buy-and-hold benchmark
    # beside a disclosure saying the difference was attributable to the rule.
    #
    # A caveat under the number is not enough, and this is the one place that
    # can be certain of it: people remember $5,160 and forget the sentence
    # beneath it. So the number is not produced.
    # Before the price-history check, and unconditionally. Ordered the other
    # way, a plan naming an instrument we cannot price reported the data gap —
    # so the user corrects the ticker, the gap closes, and the reward for
    # fixing it is a figure for a rule that still never ran. The same ordering
    # argument as `historical_lots.detect`: the unconditional refusal goes
    # first, because the conditional one reads as the whole problem.
    if scenario.event_program and not scenario.is_event_funded:
        # A rule this build still cannot execute. The guard stays for exactly
        # that case rather than being deleted along with the defect it caught:
        # the next unsupported condition must refuse rather than silently
        # replay buy-and-hold, which is what happened last time.
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None,
                "strategy_not_executed": True,
                "coverage": None, "unavailable": STRATEGY_NOT_EXECUTED}

    if not tradeable or not flows:
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None,
                "unavailable": (
                    "No price history for "
                    f"{', '.join(assets) or 'the instruments named'} over this "
                    "period, so the scenario cannot be replayed. This is a data "
                    "gap, not a result."
                )}

    # Every declared behaviour the engine cannot honour must reach the result's
    # modelling scope. Representing `dividend_policy` without saying it is not
    # simulated would move the defect one layer up rather than closing it: the
    # scenario would look enforced and the figure would silently ignore it.
    scope = declare_unsimulated(scenario, scope)

    # Pin what this run actually used, before it runs. Left empty, the
    # classifier compares two absences — and before classifier@2 reported them
    # equal, so a stored verdict claimed the account treatment was checked when
    # nothing was. An unpinnable runtime becomes a declared limitation on the
    # result rather than a blank, which is why this must happen before the scope
    # is handed to `simulate` rather than after.
    snapshot = f"prices@{sessions[-1].date()}"
    pins = pins_for(scenario, snapshot=snapshot)
    if pins.unpinned:
        scope = {**scope, "unpinned_runtimes": pins.limitations()}

    # What the engine was actually given, digested at the moment it is given.
    #
    # The delivery digest proves the resolver returned these rows. It cannot
    # prove that nothing between here and `simulate` dropped, reordered,
    # mutated or substituted them — so this closes the remaining span by
    # digesting the frame at the call itself:
    #
    #     resolved_frame_digest -> execution_input_digest -> result
    #
    # Equal digests mean the transformation was the identity. They are equal
    # today because `simulate` receives the delivered frame unchanged and the
    # instrument selection travels in `program` rather than in the data. If a
    # future path slices or adjusts the frame, this records that it did, and
    # the difference must then name a declared, versioned transformation rather
    # than pass unnoticed.
    from ..market_data.access_event import frame_digest

    execution_input = frame_digest(prices)
    # The program the scenario describes, not the one this line used to
    # hardcode. `buy_and_hold(tradeable)` ignored both the stated split and the
    # rebalancing cadence, so "60/40, rebalanced annually" replayed as equal
    # dollars in and nothing ever sold — a different strategy, returning a
    # figure under the name of the one asked for.
    allocation = scenario.allocation_rule
    holdings_policy = scenario.holdings_policy
    cadence = getattr(holdings_policy, "rebalancing_cadence", "")
    if cadence and not holdings_policy.rebalancing_allowed:
        # Two declarations that contradict each other. Refusing is the only
        # honest move: running either reading silently picks one of two
        # different strategies on the user's behalf.
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None, "strategy_not_executed": True,
                "coverage": None,
                "unavailable": (
                    f"this plan states a {cadence} rebalancing cadence and "
                    "also states that rebalancing is not allowed; the two "
                    "cannot both be honoured")}

    program = weighted(tradeable,
                       weights=getattr(allocation, "weights", None),
                       rebalance=cadence,
                       sessions=prices.index if cadence else None)
    result = simulate(prices, flows=flows, program=program,
                      cash_policy=policy, modelling_scope=scope)

    # The ledger, and then the check that it and the result agree.
    #
    # Built from the funding policy and the fills; the totals come from the
    # portfolio path. Two independent descriptions of one run, and a
    # disagreement means one of them is wrong — which is the state that shipped
    # undetected, because nothing performed this comparison. A user noticed the
    # arithmetic instead.
    execution_ledger = reconciliation = None
    if events is not None:
        from ..mission import ledger as ledger_module

        execution_ledger = ledger_module.build(
            events=events, fills=result.path.fills,
            signals=ledger_signals, unexecutable=unexecutable,
            ending_cash=float(result.path.cash.iloc[-1])
            if len(result.path.cash) else 0.0)
        reconciliation = ledger_module.reconcile(execution_ledger, result)
        if not reconciliation.agrees:
            # No figure rather than a figure with a warning. The reconciliation
            # exists to catch the engine doing something other than the plan,
            # and a result shown beside "these do not agree" is a result people
            # will quote.
            return {"result": None, "benchmarks": [], "payload": None,
                    "comparability": None, "strategy_not_executed": True,
                    "ledger": execution_ledger, "reconciliation": reconciliation,
                    "unavailable": (
                        "This result is unavailable. The executed purchases "
                        "and the reported totals do not agree, so no figure is "
                        "shown. " + "; ".join(
                            reconciliation.detail[name]
                            for name in reconciliation.failures()))}
    if access.access_event is not None:
        scope = {**scope, "execution_input_digest": execution_input,
                 "execution_input_matches_delivery":
                     execution_input == access.access_event.frame_digest}
        result = dataclasses_replace(result, modelling_scope=scope)
    # Attached here rather than by each caller. `_run` is the only place that
    # knows which access produced these figures, and a caller that has to
    # remember to attach it is a caller that can forget — which is how the live
    # path stored unattributable runs while the provenance sat one frame away.
    result = dataclasses_replace(result, market_data=access.provenance)
    if execution_ledger is not None:
        # On the result, so a stored run carries the evidence for its own
        # figure. A ledger held only in the response would make a saved run
        # unverifiable the moment the page closed — the same reason a run
        # records its market-data provenance rather than the caller attaching
        # it afterwards.
        scope = {**scope,
                 "rule_events": len(execution_ledger.rows),
                 "execution_ledger": execution_ledger.to_json(),
                 "reconciliation": reconciliation.to_json()}
        result = dataclasses_replace(result, modelling_scope=scope)
    specs = _benchmark_specs(prices, tradeable)
    benchmarks = compare(prices, flows=flows, cash_policy=policy, benchmarks=specs)

    conditions = RunConditions(
        **pins.as_conditions(),
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=policy.annual_rate,
        tax_treatment=scenario.tax_treatment,
        cost_bps=10.0, execution_lag=1,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=snapshot,
        # `None` rather than `True`: this engine executes no event program at
        # all, so a plan with no rule has nothing unexecuted, and one with a
        # rule no longer reaches here. Stating `True` would be the engine
        # certifying its own behaviour — exactly the claim the classifier
        # exists to stop resting on assertion.
        # Reaching this line *is* the evidence. A failing reconciliation
        # returns above with no result at all, so `agrees` cannot be false
        # here — and the earlier version tested it anyway, which read like a
        # control and could not fail. A mutation asserting `True`
        # unconditionally passed every test, which is what an unfalsifiable
        # guard looks like from the outside.
        #
        # The enforcement is the early return. Restating it here would be a
        # second control that can drift from the first, and the one nobody
        # reaches is the one that rots.
        declared_rule_executed=(True if scenario.event_program else None),
    )
    # Declared-to-executed coverage, and the gate.
    #
    # A figure may exist only when every material declared element either ran
    # with evidence, or was excluded by the user. Three prompts once returned
    # an identical $103,393 while each quietly dropped a different declared
    # element — a three-month period, a monthly contribution, a sell leg — and
    # each omission was individually defensible while the shared result was
    # not. This asks the general question rather than guarding each element.
    coverage = coverage_module.assess(
        scenario, stated_text=stated_text, resolved_window=resolved_window,
        frame_sessions=len(sessions), ledger=execution_ledger,
        excluded_items=[one.item for one in
                        (scenario.provenance.excluded or ())])
    if not coverage.publishable:
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None, "coverage": coverage,
                "ledger": execution_ledger, "reconciliation": reconciliation,
                "strategy_not_executed": True,
                "unavailable": coverage.refusal()}

    # One verdict per benchmark, computed here and stored with the run. The
    # worksheet reads them; it never recomputes. Every benchmark shares the
    # strategy's flows, period and account treatment by construction — `compare`
    # runs them under identical conditions — so only the allocation rule differs.
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
        "coverage": coverage,
        "ledger": execution_ledger,
        "reconciliation": reconciliation,
        # Carried out so the page can state the period it reported on. It was
        # computed here, used to slice the frame, and then discarded — so the
        # one screen that shows a figure could not say what span the figure
        # covered, which is the first thing to check when a stated period is
        # not honoured.
        "resolved_window": resolved_window,
        "unavailable": None,
    }

