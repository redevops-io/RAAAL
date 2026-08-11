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

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Sequence

from runtime_contracts import Author, VerifiedIntent

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

#: Dimensions that legitimately do not shape the executable plan.
#:
#: `objective` says what the person wants to find out. It is classified in the
#: manifest — `assess_withdrawal` and the rest are refused there — and it does
#: not belong in a `ScenarioSpecification`. Declared rather than inferred: a
#: dimension omitted from this set and from every builder refuses, which is the
#: safe direction.
NOT_EXECUTABLE = frozenset({"objective"})

#: Dimensions the scenario builders read straight off `intent.fields` rather
#: than through the `value` closure. Named here so the stranded-dimension check
#: sees both access paths; `tests/test_stranded_dimensions.py` asserts this
#: stays in step with the module rather than drifting into a stale list.
READ_DIRECTLY = frozenset({"assets", "trigger_semantics", "observed_assets"})


def _read_directly(intent: Any) -> set:
    return {name for name in READ_DIRECTLY if name in intent.fields}

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


#: Dimensions read as numbers. Kept beside `_decimal` rather than inline at the
#: call sites, because the property — a stated figure that cannot be read is
#: refused, never defaulted — has to hold for every one of them or it is not a
#: property. A numeric dimension added later and left out of this set is a
#: silent default waiting to happen; `test_every_numeric_dimension_is_listed`
#: is what makes that omission fail.
NUMERIC = frozenset({"amount", "moving_average_window"})

#: How the members of a set-valued dimension are separated.
#:
#: This split rule lives twice: here, and in `same_value` in Discovery's fusion,
#: which uses it to decide whether two readers named the same holdings. It has
#: to, because Mission may not import Discovery — the boundary is the point of
#: this module. Duplicated deliberately and pinned by a cross-layer test rather
#: than left to coincidence.
#:
#: Only one of the two copies knew about `and`, and the consequence was not a
#: near-miss. "split equally between VTI and BND" compiled to a single holding
#: named `"VTI and BND"` — one instrument, with a name no market has, weighted
#: at 100%. Fusion had agreed the sentence named two assets; Mission then built
#: a portfolio of one. `AllocationRule.canonical_form` sorts its assets, so the
#: sort ran over a one-element list and reported nothing wrong.
SET_SEPARATOR = re.compile(r"[,;]|\band\b")


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

    # A number that was stated and cannot be read is not a number that was left
    # out, and the two had the same consequence here: both call sites below
    # wrote `_decimal(...) or <default>`, so an unparseable figure fell through
    # to the default with nothing saying it had.
    #
    # "invest $1k monthly into VTI" compiled with `amount = 0`. Every other
    # field was right — VTI, monthly, first session — so the plan looked
    # entirely like the one that was asked for, and it contributed nothing. The
    # benchmark surfaced it only as a digest that moved against `$1,000`, which
    # undersold it: the finding is not that the two plans differ, it is that one
    # of them invests zero and says so nowhere.
    #
    # Refusing is deliberately the whole fix. Teaching `_decimal` about `k` and
    # `m` would close this sentence and leave the class open for the next
    # notation somebody writes, which is the same mistake as closing rotation by
    # adding `rotate` to a lemma set.
    # A recurring contribution of nothing.
    #
    # Found in the harvested corpus, and the only attested sentence of 29 that
    # reached a plan at all: "putting a portion of my cash savings into I-Bonds
    # every year" compiled to an executable plan holding I-Bonds on an annual
    # cadence with `amount = 0`, no question asked, and `amount` not even listed
    # among the applied defaults. The person named a quantity — "a portion" —
    # and was shown a plan that contributes nothing.
    #
    # The incoherence is between the two halves and needs no reading of the
    # sentence to see: the cadence says money moves every year and the amount
    # says none does. `once` is the exception and a real one — a plan may model
    # opening capital with no contributions after it.
    #
    # This is the general case of the `$1k` defect above. That one was a figure
    # stated and unreadable; this one is a figure implied and never settled.
    # Both produced a plan indistinguishable from the one asked for except that
    # it invested nothing.
    cadence = intent.fields.get("cadence")
    amount = intent.fields.get("amount")
    recurring = cadence is not None and str(cadence.value) not in ("once", "")
    # Absent, and *only* absent.
    #
    # The first version of this check read `amount is None or figure == 0`,
    # which refused an explicitly stated zero as though it were missing. Those
    # are different statements and the difference is the whole point: zero is a
    # substantive instruction — model this plan with no contributions — and
    # missing is the absence of one. Collapsing them made the runtime refuse a
    # thing the person had said, which is the mirror image of the defect this
    # check was added to close.
    #
    #     missing material quantity   -> unresolved
    #     explicitly zero quantity    -> zero
    #
    # A figure stated and *unreadable* is not this check's business either: the
    # numeric check below says so far more usefully, and firing both put two
    # refusals for `amount` on one plan, which reads as two problems.
    silent = amount is None
    if recurring and silent and "trigger_semantics" not in intent.fields:
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT", dimension="amount",
            detail=f"the plan contributes on a {cadence.value} cadence and "
                   "never says how much. A recurring contribution of zero is "
                   "a plan that does nothing, and it would be shown as though "
                   "it were the one you described"))

    for dimension in NUMERIC:
        stated = intent.fields.get(dimension)
        if stated is not None and _decimal(stated.value) is None:
            refusals.append(Refusal(
                kind="UNRESOLVED_INPUT", dimension=dimension,
                detail=f"{str(stated.value)!r} was stated for {dimension} and "
                       "cannot be read as a number. Substituting a default "
                       "here would produce a plan that looks like the one you "
                       "asked for and is not"))

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
    consulted: set = set()

    def value(dimension: str) -> Any:
        consulted.add(dimension)
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
    # A stated dimension that no builder read is an instruction that vanished.
    #
    # `moving_average_window` was the case that found this: "buy VTI below its
    # 200-day moving average" read the window correctly, no trigger existed for
    # it to attach to, and the window was dropped — so it compiled to the same
    # plan as "hold VTI for 200 days", a plain purchase. Discovery had told the
    # two apart perfectly; the drop happened here.
    #
    # Checked by construction rather than by a list of dimensions to remember:
    # anything added to the schema and not wired into a builder refuses instead
    # of being silently ignored.
    # `consulted` must mean every path a builder reads a field by, not only
    # the `value` closure. `_assets`, `_trigger` and the watched-asset lookup
    # read `intent.fields` directly, so the first version stranded `assets` on
    # every plan and refused thirty-one supported strategies — a check meant to
    # catch dropped instructions rejecting the ones that were honoured.
    consulted |= _read_directly(intent)

    stranded = sorted(set(intent.fields) - consulted - NOT_EXECUTABLE)
    if stranded:
        return Compiled(
            scenario=None, derivation=derivation,
            refusals=tuple(Refusal(
                kind="UNSUPPORTED_DIMENSION", dimension=name,
                stated_value=intent.fields[name].value,
                detail=("this build has nowhere to put it: it was read from "
                        "what you said and no part of the plan consumes it, "
                        "so executing would quietly drop it"))
                for name in stranded))

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
    return tuple(part.strip() for part in SET_SEPARATOR.split(str(stated.value))
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
