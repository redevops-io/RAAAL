"""`VerifiedIntent` → an executable plan, without ever seeing the sentence.

**This is the module that lets the legacy reader be deleted.** Everything else
in Phase 4 depends on it existing: while the only route to a
`ScenarioSpecification` runs through `compile_scenario(text, ...)`, the regex
compiler is load-bearing no matter how the intent was produced, and "Discovery
is authoritative" is a claim about the top of a pipeline whose bottom still
parses prose.

So the entry point takes an intent and nothing else. There is no `text`
parameter, no fallback that re-reads the utterance when a field is missing, and
no import of `compiler`. A test asserts the last of those, because a
convenience import added later would restore the dependency silently.

    VerifiedIntent  ──►  refusals ──►  MissionOutcome.UNSUPPORTED_CAPABILITY
                    └─►  ScenarioSpecification + Derivation

**Two things this deliberately will not do.**

It will not guess. A dimension the intent does not carry is absent, and absent
means the engine's declared default applies *and says so* — it does not mean
"go and look at the sentence again". `Author.DEFAULT` on the resulting field is
how a consumer tells the two apart, and it is the distinction the
`execution_timing` defect was made of.

It will not adjust. Every value the manifest cannot execute becomes a named
refusal, never the nearest runnable thing. That is the whole boundary, and the
reason this function returns refusals *alongside* a plan rather than quietly
producing a plan that answers a different question.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Sequence

from ..contracts import Author, VerifiedIntent
from .capability import Refusal, refusals_for
from .funding import EventTriggered, Scheduled, Trigger
from .scenario import (
    AllocationRule,
    FlowSchedule,
    HoldingsPolicy,
    Objective,
    ScenarioSpecification,
)
from .signals import Estimator, SignalKind

#: Stamped on every plan this module compiles. Bumped when the *mapping*
#: changes in a way that could turn one intent into a different plan.
COMPILER_VERSION = "quantify-mission@1"

#: What the engine applies when the intent is silent. Declared here, in one
#: place, so "the intent did not say" and "the engine chose" are the same
#: sentence for every dimension rather than a per-field accident.
DEFAULTS: Mapping[str, Any] = {
    "day_rule": "first_session_of_period",
    "execution_timing": "next_session_open",
    "allocation_method": "equal_weight_at_purchase",
    "moving_average_window": 200,
}
"""Silence, for dimensions the engine *executes*.

Every key here must be an executable dimension, and
`tests/test_mission_from_intent.py` asserts it. The first version of this table
also carried `dividend_policy` and `tax_treatment`, which the manifest refuses
and does not model — so the compiler was supplying a value for a dimension
nothing would act on, and reporting it as an applied default as though it had
meant something. That is the declared-but-not-executed shape in miniature,
inside the very module built to prevent it."""

#: Values the physical plan needs that carry no user meaning.
#:
#: `ScenarioSpecification` requires a `dividend_policy` and a `tax_treatment`
#: to be constructed. The engine runs on price series and computes no tax, so
#: neither changes a figure — they are recorded so two strategies can be told
#: apart, and the manifest refuses a *stated* one precisely because the user
#: would otherwise believe it did something.
#:
#: Kept apart from DEFAULTS so they are never reported as choices the intent
#: left open. Nobody left them open; they are not choices.
ENGINE_CONSTANTS: Mapping[str, Any] = {
    "dividend_policy": "reinvested",
    "tax_treatment": "NONE_APPLIED",
}

_TRIGGER_KIND = {
    "crossing_event": SignalKind.CROSSED_BELOW_MOVING_AVERAGE,
    "persistent_condition": SignalKind.BELOW_MOVING_AVERAGE,
}


class NotExecutable(ValueError):
    """The intent cannot become a plan, and the refusals say why."""

    def __init__(self, refusals: Sequence[Refusal]) -> None:
        self.refusals = tuple(refusals)
        super().__init__("; ".join(r.message for r in self.refusals))


@dataclass(frozen=True)
class Compiled:
    """A plan, what it was compiled from, and what had to be refused."""

    scenario: Optional[ScenarioSpecification]
    derivation: Mapping[str, Any]
    refusals: Sequence[Refusal] = ()
    applied_defaults: Sequence[str] = ()
    """Dimensions the intent did not carry, where a declared default applied.
    Surfaced rather than silent: a plan is only reproducible if the reader can
    see which of its values nobody asked for."""

    @property
    def executable(self) -> bool:
        return self.scenario is not None and not self.refusals


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").strip()
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def compile_intent(intent: VerifiedIntent, *, name: str = "plan",
                   version: int = 1,
                   benchmark_rule: Optional[str] = None) -> Compiled:
    """The only public entry point. Takes an intent; takes no text.

    Refuses a draft outright. An unsealed intent is one whose meaning Discovery
    has not closed, and compiling it would execute a guess — the seal exists
    precisely so a consumer need not judge that for itself.
    """
    if not intent.is_verified:
        raise NotExecutable((Refusal(
            kind="UNRESOLVED_INPUT", dimension="intent",
            detail="the intent is a draft; Discovery has not sealed it, so its "
                   "meaning is still open and compiling it would execute a "
                   "guess"),))

    declared = {n: f.value for n, f in intent.fields.items()}
    refusals = list(refusals_for(declared))
    for open_dimension in intent.blocking:
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT", dimension=open_dimension.dimension,
            detail="readers disagreed and it was not settled"))

    # A plan that holds nothing cannot be executed, and the first version of
    # this function compiled one happily: `AllocationRule(assets=())` with a
    # trigger whose subject was the empty string. Nothing downstream would have
    # priced it, but the failure would have surfaced as a data problem rather
    # than as what it is — an intent that never said what to buy.
    if not _assets(intent):
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT", dimension="assets",
            detail="the intent names nothing to hold, so there is no plan to "
                   "compile — this is a missing statement, not missing data"))

    derivation = {"compiled_from": intent.intent_hash,
                  "compiled_by": COMPILER_VERSION,
                  "intent_produced_by": intent.produced_by}

    if refusals:
        # No plan at all. A partial plan beside a refusal is the shape a caller
        # renders anyway, and then a figure exists for a request that was
        # refused.
        return Compiled(scenario=None, derivation=derivation,
                        refusals=tuple(refusals))

    applied: list = []

    def value(dimension: str) -> Any:
        found = intent.fields.get(dimension)
        if found is not None:
            return found.value
        if dimension in DEFAULTS:
            applied.append(dimension)
            return DEFAULTS[dimension]
        return None

    funding, allocation = _funding(intent, value)
    scenario = ScenarioSpecification(
        name=name,
        version=version,
        objective=Objective.REPLAY,
        event_program=(),
        flow_schedule=_schedule(intent, value, funding),
        allocation_rule=allocation,
        holdings_policy=HoldingsPolicy(
            sells_allowed=False, rebalancing_allowed=False,
            dividend_policy=str(ENGINE_CONSTANTS["dividend_policy"])),
        benchmark_set=None if benchmark_rule is None else _benchmarks(benchmark_rule),
        tax_treatment=str(ENGINE_CONSTANTS["tax_treatment"]),
        funding=funding,
    )
    return Compiled(scenario=scenario, derivation=derivation,
                    applied_defaults=tuple(sorted(set(applied))))


def _assets(intent: VerifiedIntent) -> Sequence[str]:
    """Holdings, from the intent's own words.

    Never resolved to a ticker here. "a core index fund" stays that, and the
    engine refuses to price it — which is the correct failure, because choosing
    VTI on the user's behalf is the substitution this whole boundary exists to
    prevent.
    """
    stated = intent.fields.get("assets")
    if stated is None:
        return ()
    return tuple(part.strip() for part in str(stated.value).split(",")
                 if part.strip())


def _funding(intent: VerifiedIntent, value):
    """`Scheduled` or `EventTriggered` — the authority on when money arrives."""
    condition = intent.fields.get("trigger_semantics")
    amount = _decimal(value("amount")) or Decimal("0")
    assets = _assets(intent)
    allocation = AllocationRule(
        assets=assets, weighting=str(value("allocation_method")))

    if condition is None:
        return Scheduled(cadence=str(value("cadence") or "once"),
                         amount=amount,
                         day_rule=str(value("day_rule"))), allocation

    watched = intent.fields.get("observed_assets")
    subject = (str(watched.value).split(",")[0].strip() if watched is not None
               else (assets[0] if assets else ""))
    window = value("moving_average_window")
    return EventTriggered(
        trigger=Trigger(
            subject=subject,
            window=int(_decimal(window) or DEFAULTS["moving_average_window"]),
            estimator=Estimator.SIMPLE,
            kind=_TRIGGER_KIND[str(condition.value)]),
        amount=amount,
        execution_timing=_timing(str(value("execution_timing")))), allocation


def _timing(name: str):
    from .funding import ExecutionTiming

    for member in ExecutionTiming:
        if member.value == name:
            return member
    return ExecutionTiming.NEXT_SESSION_OPEN


def _schedule(intent: VerifiedIntent, value, funding) -> FlowSchedule:
    """`flow_schedule` is `funding`'s projection, never a second opinion.

    The scenario carries both, and the two disagreeing is a contradiction the
    engine already refuses. Derived here rather than assembled independently so
    they cannot.
    """
    if isinstance(funding, EventTriggered):
        return FlowSchedule(cadence="event_triggered", amount=0.0,
                            day_rule=str(value("day_rule")))
    return FlowSchedule(cadence=funding.cadence, amount=float(funding.amount),
                        day_rule=funding.day_rule)


def _benchmarks(rule: str):
    from .scenario import BenchmarkSet

    return BenchmarkSet(generated_by_rule=rule, members=(), ordering="unordered")
