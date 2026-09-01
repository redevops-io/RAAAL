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
from dataclasses import dataclass, replace
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
from . import time_window
from .signals import Estimator, SignalKind
from .strategy_methods import strategy_capability
from ..config import UNIVERSE

#: Stamped on every plan this module compiles. Bumped when the *mapping*
#: changes in a way that could turn one intent into a different plan.
COMPILER_VERSION = "quantify-mission@1"

#: The universe a computed strategy runs over. The research strategies were
#: designed and evaluated on exactly these instruments, so a plan that selects
#: one holds them all: naming a subset would silently run the strategy over
#: fewer assets than it was built for and report the result under its name. A
#: strategy plan that names no holdings is completed with this rather than
#: refused, and `assets` is declared in `applied_defaults` so the substitution
#: is visible on the page like any other default.
UNIVERSE_TICKERS = tuple(asset.ticker for asset in UNIVERSE)

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
#: `stated_weights` joins them because `_weights` reads `intent.fields` rather
#: than the `value` closure — it needs the bound form the derived reader wrote,
#: not a defaulted one. Without it here the split reached the allocation rule
#: and the stranded check refused the plan anyway, reporting that nothing
#: carried an instruction it had just carried.
READ_DIRECTLY = frozenset({"assets", "trigger_semantics", "observed_assets",
                           "stated_weights"})


def _read_directly(intent: Any) -> set:
    return {name for name in READ_DIRECTLY if name in intent.fields}


def _selects_strategy(intent: Any) -> bool:
    """Whether the intent's allocation method names a computed strategy.

    Read straight off the field rather than through the `value` closure, because
    it is asked before that closure exists — `_assets` needs it to decide
    whether an intent that names no holdings is refused or completed with the
    strategy's universe.
    """
    stated = intent.fields.get("allocation_method")
    return stated is not None and strategy_capability(stated.value) is not None

#: What the engine applies when the intent is silent. Declared here, in one
#: place, so "the intent did not say" and "the engine chose" are the same
#: sentence for every dimension rather than a per-field accident.
DEFAULTS: Mapping[str, Any] = {
    "day_rule": "first_session_of_period",
    "execution_timing": "next_session_open",
    "allocation_method": "equal_weight_at_purchase",
    "moving_average_window": 200,
    # Both of these were applied silently and are now declared, which is the
    # only difference that matters: `applied_defaults` is what a reader of the
    # plan sees, and a default missing from it is a decision nobody can find.
    #
    # `cadence` fell out of `_funding` as `value("cadence") or "once"`. An
    # intent stating only assets and an amount compiled to a plan that runs
    # once, reported `('allocation_method', 'day_rule')`, and said nothing
    # about the schedule at all.
    "cadence": "once",
    # `amount` fell out of `_funding` as `_decimal(...) or Decimal("0")`.
    # Declared rather than refused, because a one-off plan with no
    # contributions is a real thing to model — opening capital and nothing
    # after it. What was wrong was that it happened without saying so, on a
    # page whose whole job is to show which values nobody asked for. A
    # *recurring* plan with no amount is still refused above: a cadence that
    # moves money every month and an amount of zero are two halves that
    # contradict each other.
    "amount": "0",
    # `dividend_policy` was an ENGINE_CONSTANT described as carrying no user
    # meaning and changing no figure. That stopped being true when the engine
    # gained a total-return path: `run_boundary._reinvests` reads this field to
    # choose which price series is resolved, and the choice is now recorded in
    # the delivery record. A value that selects the data is not a constant.
    "dividend_policy": "reinvested",
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
    "tax_treatment": "NONE_APPLIED",
}

_TRIGGER_KIND = {
    "crossing_event": SignalKind.CROSSED_BELOW_MOVING_AVERAGE,
    "persistent_condition": SignalKind.BELOW_MOVING_AVERAGE,
}


class _Absent:
    """A stand-in for a field nobody stated, so a read needs no branch."""

    value = None


_absent = _Absent()


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

#: The set separator and the currency pattern used to live here, and both are
#: gone rather than moved: they are in `discovery.canonical`, on the side of the
#: boundary where reading happens. What arrives here is comma-separated and
#: already a number. `TestTheSetMemberRuleLivesInOnePlaceNow` asserts this
#: module has no separator, because a second copy is how the first defect
#: happened — only one of them knew about `and`.


def _decimal(value: Any) -> Optional[Decimal]:
    """A canonical figure as a `Decimal`, or `None`.

    Strict on purpose. It used to strip currency words, commas and symbols —
    "1000 usd", "$1,000" — which is notation parsing, and notation is language.
    `discovery.canonical` does that now and refuses what it cannot read, so
    anything arriving here has already been settled as a number. A lenient
    parse at this point would be a second opinion about a closed question, and
    the only values it could newly accept are ones Discovery decided against.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value).strip())
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

    # Polarity arrives settled, as a policy rather than as a span.
    #
    # "I put $500 a month into VTI and never sold any of it" was refused by
    # name for `sell_action` on a build whose entire behaviour is buying and
    # never selling: the reader extracted the span correctly, and `decide()`
    # refuses any value of a REFUSED dimension, so the polarity never reached
    # it. The person described this build exactly and was told it could not be
    # run.
    #
    # Deciding that "never sold any of it" denies the disposal is a question
    # about meaning, so `discovery.canonical` answers it and seals
    # `sells_allowed=False` instead of the disposal. Nothing is refused here
    # because there is nothing left to refuse — the sentence agrees with the
    # engine rather than asking anything of it.
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
    # A canonical cadence is one of six names, so this compares values rather
    # than spellings. It used to test the reader's own words against string
    # literals, which made a coherence rule depend on how a sentence happened
    # to be phrased.
    cadence = intent.fields.get("cadence")
    amount = intent.fields.get("amount")
    recurring = cadence is not None and str(cadence.value) != "once"
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

    # A rebalancing instruction whose cadence cannot be read.
    #
    # `periodic_rebalancing` is free text on the contract — "annually", "at
    # year end", "rebalance" are all valid readings of it — and only some of
    # them state how often. The engine restores a split on a calendar, so a
    # cadence it cannot place is a plan it cannot run.
    #
    # Refused rather than defaulted, and annual is the tempting default because
    # it is what most people mean. Choosing it would invent a schedule of sales
    # the user never described, and rebalancing sells: the difference between
    # annual and quarterly is a different set of trades, a different figure,
    # and on a monotone run a materially different outcome.
    # The unbound-split refusal lives in `capability.decide`, not here.
    # Every consumer asks that function — `executable_check`, the family
    # sweep, the pilot — and a rule living in this one caller is a rule the
    # others do not have. It also stopped this compiler emitting a second
    # refusal for the same fact, which reads on the page as two problems.

    rebalancing = intent.fields.get("periodic_rebalancing")
    if rebalancing is not None and not _rebalancing_cadence(rebalancing.value):
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT", dimension="periodic_rebalancing",
            detail=f"{str(rebalancing.value)!r} asks for rebalancing without "
                   "saying how often. This build restores the split on a "
                   "calendar — annual, quarterly, monthly, weekly or biweekly "
                   "— and picking one for you would invent a schedule of sales "
                   "you did not describe"))

    for dimension in NUMERIC:
        stated = intent.fields.get(dimension)
        if stated is not None and _decimal(stated.value) is None:
            refusals.append(Refusal(
                kind="UNRESOLVED_INPUT", dimension=dimension,
                detail=f"{str(stated.value)!r} was stated for {dimension} and "
                       "cannot be read as a number. Substituting a default "
                       "here would produce a plan that looks like the one you "
                       "asked for and is not"))

    # More than one watched asset, and a trigger that observes one.
    #
    # `_funding` took the first of the list, silently. Watching SPY when
    # somebody named SPY and QQQ is a different plan, and choosing by position
    # is the substitution this boundary exists to prevent.
    watched = intent.fields.get("observed_assets")
    if watched is not None and "trigger_semantics" in intent.fields:
        members = [part for part in str(watched.value).split(",") if part.strip()]
        if len(members) > 1:
            refusals.append(Refusal(
                kind="UNSUPPORTED_DIMENSION", dimension="observed_assets",
                stated_value=watched.value,
                detail=f"{len(members)} assets are named as the thing to watch "
                       "and this build's trigger observes one. Picking one for "
                       "you would run a different rule under the same name"))

    # A plan that says it never sells and asks to be rebalanced.
    #
    # Rebalancing is the one thing that makes this build sell, so the two
    # statements contradict each other and only the person who wrote them can
    # say which they meant. Refused rather than resolved: honouring the
    # negation would drop a rebalancing schedule they described, and honouring
    # the rebalancing would sell holdings they said they never sell.
    #
    # This is the enforcement half of the split. Deciding that "never sold any
    # of it" denies the disposal is meaning and happens in Discovery; deciding
    # what a plan may therefore do is execution and happens here.
    declared_sells = intent.fields.get("sells_allowed")
    rebalances = bool(_rebalancing_cadence(
        (intent.fields.get("periodic_rebalancing") or _absent).value))
    if declared_sells is not None and not declared_sells.value and rebalances:
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT", dimension="sells_allowed",
            detail="this plan says it never sells and also asks to be "
                   "rebalanced, and rebalancing is what makes this build sell. "
                   "Both were read from what you wrote and only you can say "
                   "which one you meant"))

    # A stated execution timing this build cannot perform.
    #
    # `_timing` answered anything it did not recognise with
    # `next_session_open`. That fires on a *stated* value, so somebody asking
    # for an execution this build has no path for was given a different one
    # with nothing saying so.
    stated_timing = intent.fields.get("execution_timing")
    if stated_timing is not None:
        try:
            _timing(str(stated_timing.value))
        except UnsupportedValue:
            from .funding import ExecutionTiming

            refusals.append(Refusal(
                kind="UNSUPPORTED_DIMENSION", dimension="execution_timing",
                stated_value=stated_timing.value,
                detail=f"{str(stated_timing.value)!r} is not an execution this "
                       "build performs. It runs "
                       + ", ".join(m.value for m in ExecutionTiming)))

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

    # A strategy plan that named no holdings has its universe supplied by
    # `_assets`; declared here so the page shows `assets` as a value the engine
    # chose, not one the person did.
    if "assets" not in intent.fields and _selects_strategy(intent):
        applied.append("assets")

    funding, allocation = _funding(intent, value)

    # Read before it is used, never inside the `and`.
    #
    # Written as `bool(rebalances) and value("sells_allowed") is not False`,
    # Python short-circuits: on the common plan, which does not rebalance, the
    # left side is False and `value` is never called — so `sells_allowed` was
    # never marked consulted and the stranded check refused the very sentence
    # sealing it was meant to let through. "and never sold any of it" came back
    # refused for a dimension the user never named.
    denied_selling = value("sells_allowed") is False
    permits_selling = (bool(_rebalancing_cadence(value("periodic_rebalancing")))
                       and not denied_selling)
    scenario = ScenarioSpecification(
        name=name,
        version=version,
        objective=Objective.REPLAY,
        event_program=(),
        flow_schedule=_schedule(intent, value, funding),
        allocation_rule=allocation,
        # Rebalancing is the one thing that makes this build sell, so the two
        # permissions follow from whether a cadence was stated rather than
        # being hardcoded off. A plan that rebalances and forbids selling is
        # incoherent, and it is refused above rather than resolved here.
        #
        # `value("sells_allowed")` is read for its own sake: a sealed policy
        # that no builder consumed would be stranded, and the stranded check
        # would refuse the very sentence — "and never sold any of it" — that
        # sealing it was meant to let through. Found by compiling that sentence
        # end to end; the suite had no case that did.
        holdings_policy=HoldingsPolicy(
            sells_allowed=permits_selling,
            rebalancing_allowed=bool(_rebalancing_cadence(value("periodic_rebalancing"))),
            rebalancing_cadence=_rebalancing_cadence(value("periodic_rebalancing")),
            dividend_policy=str(value("dividend_policy"))),
        benchmark_set=None if benchmark_rule is None else _benchmarks(benchmark_rule),
        tax_treatment=str(ENGINE_CONSTANTS["tax_treatment"]),
        funding=funding,
    )

    # The stated evaluation window, carried onto the scenario so the engine
    # restricts the replay to it.
    #
    # `capability.decide` (via `refusals_for` above) has already refused an
    # open-ended, explicit-range or rolling window, so a value reaching an
    # executable compile is a trailing duration the engine resolves — "the past
    # 5 years". Set on the provenance the engine reads (`_resolve_window` ->
    # `time_window.resolve` slices the snapshot to it), and the figure is then
    # reported over exactly that period rather than the whole history. `value`
    # marks it consulted so the stranded check below does not read a consumed
    # instruction as a dropped one. Guarded again here rather than trusting the
    # upstream refusal: a value can arrive by a path `decide` did not see, and a
    # window set without a resolvable duration would slice to an empty frame.
    # `evaluation_period` is the window read from "over the past 5 years";
    # `holding_period` is what the model reads a plan's own "for 5 years" as,
    # and on a build that holds to the end of the evaluation period the two name
    # the same trailing period (capability.decide lets a trailing duration run
    # under either). Prefer the window when both are present; consult both by
    # name so the stranded check does not read the one not chosen as dropped.
    stated_period = value("evaluation_period")
    holding_period = value("holding_period")
    stated_window = stated_period if stated_period is not None else holding_period
    if stated_window is not None:
        window = time_window.from_canonical(str(stated_window))
        if window is not None and window.supported and (window.years or window.months):
            scenario = replace(scenario, provenance=replace(
                scenario.provenance, time_window=window))

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
    """Holdings, read from the canonical comma-separated form.

    Never resolved to a ticker here. "a core index fund" stays that, and the
    engine refuses to price it — which is the correct failure, because choosing
    VTI on the user's behalf is the substitution this whole boundary exists to
    prevent.

    Splitting on an English "and" used to happen here, from a copy of a rule
    that also lived in Discovery, and only one copy knew the word. "split
    equally between VTI and BND" compiled to a single holding named `"VTI and
    BND"`. One copy now, in `discovery.canonical`, and what arrives here is a
    list.
    """
    stated = intent.fields.get("assets")
    if stated is None:
        # A computed strategy carries its own universe; a plan that selects one
        # but names no holdings is completed with it rather than refused. The
        # declaration of this substitution is added to `applied_defaults` by the
        # caller, so it reads on the page as a default like any other.
        if _selects_strategy(intent):
            return UNIVERSE_TICKERS
        return ()
    return tuple(part.strip() for part in str(stated.value).split(",")
                 if part.strip())


def _weights(intent: VerifiedIntent) -> Dict[str, Decimal]:
    """A stated split, attached to the holdings it belongs to.

    Reads only the bound form `TICKER=weight,TICKER=weight` that
    `discovery.derived_readers.weight_binding` authors from the sentence. A
    bare ratio is deliberately not accepted: `60/40` beside two instruments
    could mean either assignment, and picking one runs 40/60 under the name
    60/40 — a wrong executable meaning, which is worse than the refusal it
    would replace.
    """
    stated = intent.fields.get("stated_weights")
    if stated is None:
        return {}

    weights: Dict[str, Decimal] = {}
    for part in str(stated.value).split(","):
        name, _, figure = part.partition("=")
        amount = _decimal(figure)
        if not name.strip() or amount is None:
            return {}
        weights[name.strip()] = amount
    return weights if len(weights) > 1 else {}


def _rebalancing_cadence(stated: Any) -> str:
    """The canonical cadence a rebalancing instruction carries, or "".

    This used to normalise free text by importing `discovery.syntax`. Its
    argument was right — one place must decide what "annually" means, or two
    users get different schedules from the same sentence — and the place is
    Discovery, which now seals the canonical name. What is left is a read.
    """
    return "" if stated in (None, "") else str(stated)


def _funding(intent: VerifiedIntent, value):
    """`Scheduled` or `EventTriggered` — the authority on when money arrives."""
    condition = intent.fields.get("trigger_semantics")
    amount = _decimal(value("amount")) or Decimal("0")
    assets = _assets(intent)
    allocation = AllocationRule(
        assets=assets, weighting=str(value("allocation_method")),
        weights={k: float(v) for k, v in _weights(intent).items()})

    if condition is None:
        # `value("cadence")` and nothing else. The `or "once"` that used to sit
        # here supplied a schedule without declaring one; `cadence` is in
        # DEFAULTS now, so the same value arrives through the path that reports
        # it.
        return Scheduled(cadence=str(value("cadence")),
                         amount=amount,
                         day_rule=str(value("day_rule"))), allocation

    watched = intent.fields.get("observed_assets")
    subject = (_watched(watched) if watched is not None
               else (assets[0] if assets else ""))
    window = value("moving_average_window")
    return EventTriggered(
        trigger=Trigger(
            subject=subject,
            window=int(_decimal(window)),
            estimator=Estimator.SIMPLE,
            kind=_TRIGGER_KIND[str(condition.value)]),
        amount=amount,
        execution_timing=_timing(str(value("execution_timing")))), allocation


def _watched(stated) -> str:
    """The single asset a trigger observes.

    Taking the first of a stated list used to happen here, silently. Which
    asset is watched is meaning, and a plan that watches SPY while the person
    named SPY and QQQ is a different plan — so more than one is refused
    upstream by `compile_intent` rather than resolved by position.
    """
    members = [part.strip() for part in str(stated.value).split(",")
               if part.strip()]
    return members[0] if members else ""


class UnsupportedValue(ValueError):
    """A stated value this build has no execution for."""


def _timing(name: str):
    """The declared timing, or a refusal.

    It used to return `NEXT_SESSION_OPEN` for anything it did not recognise —
    a substitution on a *stated* value, which is worse than one on an absent
    field: somebody asked for an execution this build cannot perform and was
    given a different one with no sign of it.
    """
    from .funding import ExecutionTiming

    for member in ExecutionTiming:
        if member.value == name:
            return member
    raise UnsupportedValue(name)


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
