"""Calculation, and the decision whether its answer may be published.

`_run` did four different jobs and its name said none of them. Classified
before anything moved:

    EVALUATION        eligible sessions, signals, contributions, orders and
                      fills, cash flows, metrics — everything that decides what
                      the number *is*
    EVALUATION-GATE   reconciliation and coverage — the checks that decide
                      whether the number may be shown at all
    APPLICATION       benchmarks, the comparison payload, comparability
                      verdicts, the resolved window carried out for the page
    PROVENANCE        runtime pins, the execution-input digest, the market-data
                      record attached to the result

This module is the first two. The rest stayed where it was, because moving
benchmark and presentation logic into an evaluation service would take the
application with it — and none of it changes the figure or whether the figure is
publishable.

**The exit condition it exists for**: given a plan, a frame and a policy, this
produces the seven streams and the publish/refuse decision without importing
`workspace`. That is checkable, and `tests/test_evaluation_boundary.py` checks
it.

**Why the gate is here and not in the application.** A refusal is a result. The
reconciliation catches the engine doing something other than the plan, and
coverage catches a declared element that quietly did not run — three prompts
once returned an identical figure while each dropped a different declared
element. Both must be able to withhold the number, and a gate the caller can
forget to ask is a gate that will be forgotten.

**What stays outside, deliberately.** Runtime pins and the execution-input
digest are provenance: they travel with the result and do not decide it. They
also need the sliced frame, which only exists once the window is resolved — so
the caller supplies `pin_scope`, called with the sessions this run will use,
rather than this module reaching for the application's pinning machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..mission import coverage as coverage_module
from ..mission import ledger as ledger_module
from ..mission.accounting import CashFlow, CashPolicy
from ..mission.rebalance import weighted
from ..mission.scenario import UNSIMULATED
from ..mission.schedule import UnsupportedCadence
from ..mission.simulate import simulate

#: What the engine says when a declared rule is not one it executes.
#:
#: Moved here verbatim rather than reworded. It is the sentence a user reads,
#: two tests assert it, and a decomposition that quietly rewrote the wording
#: would be a product change wearing a refactor's clothes — which is exactly
#: what the first version of this move did.
STRATEGY_NOT_EXECUTED = (
    "This result is unavailable. The plan contains a conditional purchase "
    "rule, but this version of Quantify did not execute that rule. No strategy "
    "result is shown. Your description and clarifications remain saved."
)


@dataclass(frozen=True)
class Evaluated:
    """What the calculation produced, and whether it may be published.

    `publishable` is not derived from `result is not None`. A run can produce a
    perfectly good `MissionResult` and still be unpublishable — a failing
    reconciliation and an incomplete coverage assessment both do exactly that —
    and collapsing the two would let a figure through on the strength of having
    been computed.
    """

    publishable: bool
    refusal: Optional[str] = None
    result: Optional[Any] = None
    ledger: Optional[Any] = None
    reconciliation: Optional[Any] = None
    coverage: Optional[Any] = None
    resolved_window: Optional[Any] = None
    prices: Optional[Any] = None
    """The frame actually handed to the engine, after the window was applied.
    Returned so the caller can digest what was executed on rather than what was
    delivered — the two differ the moment a window slices, and the difference is
    the thing the digest exists to record."""
    flows: tuple = ()
    tradeable: tuple = ()
    scope: Optional[Dict[str, Any]] = None


def _refused(why: str, **carried) -> Evaluated:
    return Evaluated(publishable=False, refusal=why, **carried)


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


def _flows_from(schedule, sessions: pd.DatetimeIndex) -> List[CashFlow]:
    """Turn a declared schedule into dated contributions.

    Delegates to `mission.schedule.expand`. Kept as a named function here
    because it is the route's seam and several tests reach for it directly.
    """
    from ..mission.schedule import expand

    return expand(schedule, sessions, cash_flow=CashFlow)


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


def evaluate_plan(scenario, prices, *, scope: Optional[Dict[str, Any]] = None,
                  stated_text: str = "",
                  pin_scope: Optional[Callable[[Any], Dict[str, Any]]] = None
                  ) -> Evaluated:
    """One plan, one frame, one answer — or one refusal that says why.

    Every early return here is a decision that no figure may exist. They are
    ordered, and the order is load-bearing: the unconditional refusals come
    before the conditional ones, because a plan naming an instrument we cannot
    price used to report the data gap first — so the user corrected the ticker,
    the gap closed, and the reward was a figure for a rule that still never ran.
    """
    resolved_window = _resolve_window(scenario, prices)
    if resolved_window is not None:
        first = resolved_window.warmup_start or resolved_window.start
        prices = prices.loc[str(first):str(resolved_window.end)]

    sessions = prices.index
    policy = CashPolicy.idle()
    assets = list(scenario.allocation_rule.assets)
    # Custom indices (e.g. VT) carry no series of their own — the data service
    # named their components and delivered them; the valuation engine computes
    # the index level from those columns here, before pricing, so a plan that
    # holds one trades a real column rather than reading as no price history.
    from .index_calc import materialise
    prices = materialise(prices, assets)
    tradeable = [one for one in assets if one in prices.columns]

    events, ledger_signals, unexecutable, funding_error = _funding_events(
        scenario, prices, sessions)
    if funding_error is not None:
        return _refused(funding_error, resolved_window=resolved_window)

    try:
        flows = (tuple(CashFlow(date=event.session, amount=float(event.amount))
                       for event in events)
                 if events is not None
                 else _flows_from(scenario.flow_schedule, sessions))
    except UnsupportedCadence as refused:
        return _refused(str(refused), resolved_window=resolved_window)

    if scenario.event_program and not scenario.is_event_funded:
        return _refused(STRATEGY_NOT_EXECUTED, resolved_window=resolved_window)

    if not tradeable or not flows:
        return _refused(
            "No price history for "
            f"{', '.join(assets) or 'the instruments named'} over this period, "
            "so the scenario cannot be replayed. This is a data gap, not a "
            "result.", resolved_window=resolved_window)

    scope = declare_unsimulated(scenario, scope)
    if pin_scope is not None:
        scope = {**scope, **pin_scope(sessions)}

    holdings_policy = scenario.holdings_policy
    cadence = getattr(holdings_policy, "rebalancing_cadence", "")
    if cadence and not holdings_policy.rebalancing_allowed:
        return _refused(
            f"this plan states a {cadence} rebalancing cadence and also states "
            "that rebalancing is not allowed; the two cannot both be honoured",
            resolved_window=resolved_window)

    # A computed allocation (risk parity, momentum, …) restores to weights a
    # research strategy produces each period, rather than to a stated split. It
    # needs a calendar to rebalance on: a strategy that never rebalances is not
    # the strategy, so an unscheduled one is refused rather than run as a plain
    # buy-and-hold wearing the strategy's name.
    from ..mission.strategy_methods import strategy_capability
    capability_id = strategy_capability(
        getattr(scenario.allocation_rule, "weighting", ""))
    if capability_id is not None:
        if not cadence:
            return _refused(
                "this plan allocates by a computed strategy, which restores its "
                "weights on a calendar. Naming a monthly, quarterly or annual "
                "rebalancing cadence would give it a schedule to run on.",
                resolved_window=resolved_window)
        from ..strategies import CAPABILITY_BY_ID
        from ..mission.rebalance import strategy_driven
        min_history = int(getattr(
            CAPABILITY_BY_ID.get(capability_id, None), "min_history", 1) or 1)
        program = strategy_driven(
            tradeable, capability_id=capability_id, cadence=cadence,
            sessions=prices.index, min_history=min_history)
    else:
        program = weighted(
            tradeable,
            weights=getattr(scenario.allocation_rule, "weights", None),
            rebalance=cadence,
            sessions=prices.index if cadence else None)
    result = simulate(prices, flows=flows, program=program,
                      cash_policy=policy, modelling_scope=scope)

    # --- the gate ---------------------------------------------------------
    execution_ledger = reconciliation = None
    if events is not None:
        execution_ledger = ledger_module.build(
            events=events, fills=result.path.fills, signals=ledger_signals,
            unexecutable=unexecutable,
            ending_cash=float(result.path.cash.iloc[-1])
            if len(result.path.cash) else 0.0)
        reconciliation = ledger_module.reconcile(execution_ledger, result)
        if not reconciliation.agrees:
            return _refused(
                "This result is unavailable. The executed purchases and the "
                "reported totals do not agree, so no figure is shown. "
                + "; ".join(reconciliation.detail[name]
                            for name in reconciliation.failures()),
                ledger=execution_ledger, reconciliation=reconciliation,
                resolved_window=resolved_window)

    coverage = coverage_module.assess(
        scenario, stated_text=stated_text, resolved_window=resolved_window,
        frame_sessions=len(sessions), ledger=execution_ledger,
        fills=getattr(result.path, "fills", ()),
        excluded_items=[one.item for one in
                        (scenario.provenance.excluded or ())])
    if not coverage.publishable:
        return _refused(coverage.refusal(), coverage=coverage,
                        ledger=execution_ledger, reconciliation=reconciliation,
                        resolved_window=resolved_window)

    return Evaluated(
        publishable=True, result=result, ledger=execution_ledger,
        reconciliation=reconciliation, coverage=coverage,
        resolved_window=resolved_window, prices=prices,
        flows=tuple(flows), tradeable=tuple(tradeable), scope=scope)
