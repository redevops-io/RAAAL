"""Natural language into an inspectable artifact.

This is the point where an unverified source is allowed into an otherwise honest
stack, so it is the strictest layer rather than the friendliest. Every hidden
choice introduced here propagates cleanly — and misleadingly — through everything
downstream, because everything downstream is correct.

**The language model is quarantined to stage 1.** Parse produces a
`ParsedUtterance`: structured, boring, and *data to be verified* rather than
decisions to be trusted. Stages 2–10 are deterministic and would produce the same
scenario from the same parse a year from now. That is what makes the compiler
auditable when the model underneath it is not.

    1  parse         text -> recognised phrases            (model, or rules)
    2  normalize     phrases -> canonical concepts         deterministic
    3  resolve       concepts -> versioned identifiers     deterministic
    4  contradict    detect conflicts in the compiled form deterministic
    5  unresolved    name every material choice left open  deterministic
    6  defaults      apply a *versioned* default set       deterministic
    7  compile       emit provisional artifacts            deterministic
    8  verify        run every check the library runs      deterministic
    9  confirm       plain-language statements, grouped    deterministic
    10 commit        mint immutable ids, begin simulation

The one rule that makes the rest work: **an unrecognised phrase becomes
`unresolved`, never a default.** A compiler that guesses when it does not know is
indistinguishable from one that knows, right up until it is wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .defaults import DEFAULT_SET, DefaultSet
from .scenario import AllocationRule, BenchmarkSet, HoldingsPolicy, ScenarioSpecification
from .spec import Contradiction, FlowSchedule, Inference, Objective, Provenance, Unresolved


class Origin(str, Enum):
    STATED = "STATED"
    """The text said it unambiguously. The user owns this."""

    INFERRED = "INFERRED"
    """A versioned default supplied it. The user must confirm it."""

    UNRESOLVED = "UNRESOLVED"
    """Nobody has decided. The scenario may run provisionally and may not save."""


@dataclass(frozen=True)
class Recognition:
    """One phrase the parser matched, and the field it determines."""

    field: str
    value: str
    span: str
    """The user's words, verbatim, so the confirmation screen can quote rather
    than paraphrase. A paraphrase is the compiler's account of what was said."""

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "value": self.value, "span": self.span}


@dataclass(frozen=True)
class ParsedUtterance:
    """The only thing a model produces. Everything after this is deterministic."""

    text: str
    recognitions: Sequence[Recognition] = ()
    assets: Sequence[str] = ()
    unrecognized: Sequence[str] = ()
    """Names that map to more than one instrument. Every entry is a key of
    `AMBIGUOUS_NAMES`, so the question can offer the actual choices."""

    unclear: Sequence[str] = ()
    """Phrases stage 1 could not place at all. Free text, and deliberately kept
    apart from `unrecognized`: "which share class?" offers options, "I could not
    read this" does not, and a screen that renders them the same way is asking
    the user to tell the two apart themselves."""

    template_hint: Optional[str] = None
    """A life-event template that covers this, if one does. The compiler hands
    off rather than paraphrasing rules the template states with citations."""

    def value_of(self, field_name: str) -> Optional[Recognition]:
        for r in self.recognitions:
            if r.field == field_name:
                return r
        return None

    def to_json(self) -> Dict[str, Any]:
        """Serialized so a saved plan can pin the parse it was compiled from.

        Revisiting a plan recompiles from this, never by asking a model again: a
        model that has changed since would silently alter a plan the user
        already confirmed and saved.
        """
        return {
            "text": self.text,
            "recognitions": [r.to_json() for r in self.recognitions],
            "assets": list(self.assets),
            "unrecognized": list(self.unrecognized),
            "unclear": list(self.unclear),
            "template_hint": self.template_hint,
        }


# --- stage 1: recognisers -------------------------------------------------
#
# Deterministic phrase rules, deliberately narrow. Each pair below distinguishes
# two readings that are economically different and that casual language does not
# separate — which is exactly where a fluent model is most likely to smooth over
# the difference, because both readings sound like the same sentence.

_RULES: Sequence[Tuple[str, str, str]] = (
    # (field, canonical value, pattern)
    ("trigger_semantics", "crossing_event",
     r"\b(only )?on the day\b[^.]*\bcross(?:es|ing|ed)?\b|\bwhen\b[^.]*\bcrosses (?:below|above)\b"),
    ("trigger_semantics", "persistent_condition",
     r"\bwhenever\b[^.]*\b(?:is |trades |closes )?(?:below|above)\b|\bwhile\b[^.]*\b(?:below|above)\b"),

    ("weighting", "equal_weight_maintained",
     r"\brebalanc\w+\b[^.]*\bequal\b|\bequal\b[^.]*\bweights?\b[^.]*\b(?:maintain|keep|rebalanc)\w*|\bkeep (?:them|the portfolio) equal\b"),
    ("weighting", "equal_weight_at_purchase",
     r"\bequal(?:ly)?[- ]?weight\w*\b[^.]*\b(?:each|every|at) purchase\b|\bequal dollars\b|\bbuy\b[^.]*\bequally\b"),

    ("funding_source", "contribution",
     r"\b(?:use|invest|with|from) (?:the |my |that |this )?(?:monthly )?contribution\b|\bout of (?:the |my |that |this )?(?:monthly |usual )?(?:contribution|paycheck|paycheque|transfer)\b"),
    ("funding_source", "additional_cash",
     r"\badditional cash\b|\bextra cash\b|\bseparate(?:ly)? from (?:the |my )?contribution\b|\bon top of\b"),

    ("earnings_timing", "earnings_date",
     r"\bon (?:the )?earnings (?:date|day)\b|\bthe day of earnings\b"),
    ("earnings_timing", "first_session_after_earnings",
     r"\bfirst (?:tradable |tradeable |trading )?(?:day|session) after earnings\b|\bafter (?:the )?earnings (?:release|announcement)\b"),

    ("dividends", "held_as_cash",
     r"\bdividends?\b[^.]*\b(?:as cash|in cash|not reinvest\w*|held as cash)\b"),
    ("dividends", "reinvested",
     r"\breinvest\w*\b[^.]*\bdividends?\b|\bdividends?\b[^.]*\breinvest\w*\b"),

    ("vesting_action", "exercise_and_sell",
     r"\bexercis\w+\b[^.]*\b(?:and )?sell\b"),
    ("vesting_action", "sell_vested_shares",
     r"\bsell\b[^.]*\bvested shares\b|\bsell (?:the )?vest\w+\b"),

    ("contribution_day_rule", "calendar_first_rolled_forward",
     r"\bfirst calendar day\b|\bon the 1st\b|\bthe first of (?:the |each |every )?month\b"),
    ("contribution_day_rule", "first_session_of_period",
     r"\bfirst trading (?:day|session)\b|\bfirst market day\b"),

    ("sells_allowed", "false", r"\bnever sell\b|\bdo(?:n'?t| not) sell\b|\bno sell(?:ing|s)?\b"),
    ("moving_average_kind", "exponential", r"\bexponential\b|\bEMA\b"),
    ("moving_average_kind", "simple", r"\bsimple\b(?:[^.]*\b(?:moving )?average\b)|\bSMA\b"),
)

#: Ordered most specific first: "every other week" must not be read as "week",
#: and "every year" must not be read as "ear". The corpus found this list
#: covering four of the nine cadences its catalog actually uses — a user writing
#: "annually", the single most common cadence in the corpus, was asked how often
#: their contribution arrives immediately after saying so.
_CADENCE = (
    ("biweekly", r"\b(?:every|each) (?:two weeks|fortnight)\b|\bbiweekly\b|\bevery other week\b|\bsemi-?monthly\b"),
    ("monthly", r"\b(?:every|each) month\b|\bmonthly\b"),
    ("weekly", r"\b(?:every|each) week\b|\bweekly\b"),
    ("quarterly", r"\b(?:every|each) quarter\b|\bquarterly\b"),
    # No bare `\bannual\b`: it is an adjective, and it matched "annual
    # rebalance" in a *benchmark* clause, so the compiler read the user's
    # contribution cadence out of a sentence about what to compare against.
    # Found by the corpus immediately after this cadence was added.
    ("annual", r"\b(?:every|each) year\b|\bannually\b|\bonce a year\b|\byearly\b"),
    ("payroll", r"\bevery pay ?day\b|\beach pay ?day\b|\bout of (?:each|every|my) pay ?che(?:ck|que)\b|\bwith each pay ?che(?:ck|que)\b|\bper pay period\b"),
    ("daily", r"\b(?:every|each) day\b|\bdaily\b"),
    ("once", r"\blump sum\b|\ball at once\b|\bone ?-?off\b"),
)

_AMOUNT = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")

#: Whether the description implies a market signal at all. Without one there is
#: no trigger, and asking how a trigger should behave invents a condition the
#: user never mentioned — which then blocks a perfectly complete plan for a
#: reason that does not exist. Only ask about what was described.
_MENTIONS_SIGNAL = re.compile(
    r"\b(?:below|above|cross(?:es|ing|ed)?|moving average|DMA|SMA|EMA|"
    r"whenever|when(?:ever)? .{0,30}\b(?:drops|falls|rises)|dips?|breaks?)\b",
    re.IGNORECASE,
)
_MENTIONS_AVERAGE = re.compile(
    r"\b(?:moving average|DMA|SMA|EMA|[0-9]{1,4}[- ]?day average)\b", re.IGNORECASE)
_TICKER = re.compile(r"\b([A-Z]{1,5})\b")

#: Language that means a life-event template exists for this, and that the
#: generic compiler should hand off rather than improvise. Inventing vesting
#: semantics from prose is exactly the hidden-choice problem: the template
#: encodes cited, checkable rules, and a paraphrase of them would not.
_TEMPLATE_HINTS = (
    ("rsu-vesting", re.compile(
        r"\b(RSUs?|restricted stock|vest(?:s|ed|ing)?|equity grant|"
        r"stock (?:award|grant))\b", re.IGNORECASE)),
)

#: Names a user will write that do not resolve to one instrument. Guessing here
#: is how a scenario silently prices the wrong security.
AMBIGUOUS_NAMES = {
    "google": ("GOOGL", "GOOG"),
    "alphabet": ("GOOGL", "GOOG"),
    "berkshire": ("BRK.A", "BRK.B"),
}


def parse(text: str) -> ParsedUtterance:
    """Stage 1. Recognise phrases; recognise nothing else.

    The narrowness is the feature. Anything this does not match becomes an
    unresolved question rather than a default, so the failure mode is an extra
    confirmation rather than a wrong number.
    """
    recognitions: List[Recognition] = []
    claimed: set = set()

    for field_name, value, pattern in _RULES:
        if field_name in claimed:
            continue
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            recognitions.append(
                Recognition(field=field_name, value=value, span=match.group(0).strip())
            )
            claimed.add(field_name)

    for cadence, pattern in _CADENCE:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and "cadence" not in claimed:
            recognitions.append(
                Recognition("cadence", cadence, match.group(0).strip()))
            claimed.add("cadence")

    amount = _AMOUNT.search(text)
    if amount:
        recognitions.append(
            Recognition("amount", amount.group(1).replace(",", ""), amount.group(0)))

    lowered = text.lower()
    unrecognized = [name for name in AMBIGUOUS_NAMES if name in lowered]

    reserved = {"SPY", "DMA", "EMA", "SMA", "RSU", "ESPP", "IRA"}
    assets = [t for t in _TICKER.findall(text) if t not in reserved and len(t) >= 2]

    hint = next((name for name, pattern in _TEMPLATE_HINTS
                 if pattern.search(text)), None)

    return ParsedUtterance(text=text, recognitions=tuple(recognitions),
                           assets=tuple(dict.fromkeys(assets)),
                           unrecognized=tuple(unrecognized),
                           template_hint=hint)


# --- stages 2-6: normalize, resolve, contradict, unresolved, defaults ------

#: Material choices, and what to ask when the text did not settle them. Every
#: entry names a consequence, because a question without one gets answered at
#: random.
_QUESTIONS: Mapping[str, Tuple[str, str]] = {
    "trigger_semantics": (
        "Should this fire on every day the condition holds, or only on the day it "
        "first becomes true?",
        "A persistent condition buys repeatedly through a downtrend; a crossing "
        "event buys once. Over a long decline these are entirely different plans.",
    ),
    "weighting": (
        "Does 'equally' mean equal dollars each time you buy, or keeping the whole "
        "position equal over time?",
        "Keeping positions equal requires selling what rose, which conflicts with "
        "never selling.",
    ),
    "funding_source": (
        "Does this buy come out of the regular contribution, or is it additional "
        "money?",
        "Taken from the contribution, the plan buys the same total; as extra "
        "money it invests more, and more money in a rising market always looks "
        "like a better rule.",
    ),
    "amount": (
        "How much are you contributing, and how often?",
        "Every figure scales with it, and the timing changes the money-weighted "
        "return even when the strategy is identical.",
    ),
    "starting_capital": (
        "How much are you starting with, before any contributions?",
        "Starting capital and contributions produce very different paths for the "
        "same rule.",
    ),
    "asset_identity": (
        "Which share class do you mean?",
        "The classes have different prices, voting rights and index membership, "
        "and are not interchangeable in a backtest.",
    ),
    "benchmark_set": (
        "What should this be compared against?",
        "A result with nothing to compare it to cannot be interpreted.",
    ),
}


@dataclass(frozen=True)
class CompilerResult:
    """The whole compilation, inspectable before anything runs."""

    intent_id: Optional[str]
    status: str
    stated: Sequence[str]
    inferred: Sequence[Inference]
    contradictions: Sequence[Contradiction]
    unresolved: Sequence[Unresolved]
    defaults_ref: str
    scenario: Optional[ScenarioSpecification]
    verification: Sequence[str] = ()
    template_hint: Optional[str] = None
    template_offer: Optional[Unresolved] = None
    """A better route exists for this description. An offer, deliberately not an
    `unresolved` entry: it does not block, and the blocking list is the one a
    reader has to be able to trust."""

    @property
    def can_simulate(self) -> bool:
        """A provisional interpretation may run so the user can see its shape.

        Only a structural contradiction stops this: a scenario that cannot
        execute as written has no shape to show.
        """
        return self.scenario is not None and self.scenario.is_runnable

    @property
    def can_save(self) -> bool:
        """Every material choice confirmed or explicitly accepted."""
        return (self.scenario is not None
                and self.scenario.provenance.is_complete
                and self.scenario.is_runnable)

    def confirmation(self) -> Dict[str, Any]:
        """Stage 9, as data. Grouped the way a reader has to check it.

        Each statement names the compiled field it controls, so two people
        reading this can agree on what will be simulated and on who decided each
        part — which is the only success criterion that matters here.
        """
        return {
            "you_stated": [
                {"statement": s, "controls": None} for s in self.stated
            ],
            "we_inferred": [
                {"statement": f"{i.field.replace('_', ' ')}: {i.value}",
                 "why": i.why, "controls": i.field, "confirmed": i.confirmed}
                for i in self.inferred
            ],
            "these_conflict": [
                {"statement": c.detail, "between": list(c.between),
                 "resolved": c.resolved}
                for c in self.contradictions
            ],
            "we_still_need": [
                {"question": u.question, "why_it_matters": u.why_it_matters,
                 "controls": u.field}
                for u in self.unresolved
            ],
            "a_better_route": (
                {"question": self.template_offer.question,
                 "why_it_matters": self.template_offer.why_it_matters,
                 "controls": self.template_offer.field}
                if self.template_offer else None),
            "defaults_ref": self.defaults_ref,
            "can_simulate": self.can_simulate,
            "can_save": self.can_save,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "status": self.status,
            "stated": list(self.stated),
            "inferred": [i.to_json() for i in self.inferred],
            "contradictions": [c.to_json() for c in self.contradictions],
            "unresolved": [u.to_json() for u in self.unresolved],
            "template_offer": (self.template_offer.to_json()
                               if self.template_offer else None),
            "defaults_applied": self.defaults_ref,
            "scenario_preview": self.scenario.to_json() if self.scenario else None,
            "verification": list(self.verification),
            "can_simulate": self.can_simulate,
            "can_save": self.can_save,
        }


def compile_scenario(
    text: str,
    *,
    name: str = "scenario",
    version: int = 1,
    intent_id: Optional[str] = None,
    defaults: DefaultSet = DEFAULT_SET,
    objective: Objective = Objective.REPLAY,
    benchmark_rule: Optional[str] = None,
    parsed: Optional[ParsedUtterance] = None,
) -> CompilerResult:
    """Stages 1–8. Deterministic from the parse onward.

    `parsed` is the injection point for stage 1. Pass one produced by
    `parse_model.parse_with_model` to widen recognition with a language model, or
    leave it out for the deterministic rules alone. Everything below this line
    behaves identically either way — which is the property that lets a model sit
    in stage 1 at all, and the reason a saved plan recompiles from its stored
    parse rather than by asking a model again.
    """
    if parsed is None:
        parsed = parse(text)
    elif parsed.text != text:
        raise ValueError(
            "the supplied parse is of different text than the one being "
            "compiled; stages 2-10 would then describe a scenario nobody wrote")

    stated: List[str] = [r.span for r in parsed.recognitions]
    inferred: List[Inference] = []
    unresolved: List[Unresolved] = []

    def settle(field_name: str, default_field: Optional[str] = None) -> Optional[str]:
        """Stated wins; otherwise a *versioned* default; otherwise a question."""
        found = parsed.value_of(field_name)
        if found:
            return found.value
        entry = defaults.get(default_field or field_name)
        if entry:
            inferred.append(Inference(field=field_name, value=entry.value,
                                      why=entry.why, confirmed=False))
            return entry.value
        question, why = _QUESTIONS.get(
            field_name, (f"What should {field_name.replace('_', ' ')} be?",
                         "This changes the result."))
        unresolved.append(Unresolved(field=field_name, question=question,
                                     why_it_matters=why))
        return None

    # Ask only about what the description implies. A question about a condition
    # nobody mentioned is noise at best, and at worst it blocks a complete plan
    # on an ambiguity that does not exist in it.
    has_signal = bool(_MENTIONS_SIGNAL.search(text))
    trigger = settle("trigger_semantics") if has_signal else None
    if has_signal and _MENTIONS_AVERAGE.search(text):
        settle("moving_average_kind")
    if has_signal:
        settle("execution_timing")

    # "Equally" only means something across more than one holding — so the
    # question is not *asked* for a single recognised asset. But a weighting the
    # user actually stated is settled either way: the guard exists to avoid
    # inventing a question, not to discard an answer.
    #
    # Found by the strategy corpus. 94% of deliberately contradictory prompts
    # went unreported because the parser recognised no ticker in phrases like
    # "invest only above floor", so a stated "rebalance to equal weights"
    # alongside "never sell" was dropped before the contradiction check ran.
    # That is declaration without behaviour, in the compiler itself.
    states_weighting = parsed.value_of("weighting") is not None
    weighting = (settle("weighting")
                 if states_weighting or len(parsed.assets) > 1 else None)

    settled_dividends = settle("dividends")
    day_rule = settle("contribution_day_rule")

    # Funding source only matters once there is both a contribution and a buy
    # rule to fund; asking otherwise is noise.
    amount_seen = parsed.value_of("amount")
    if amount_seen and trigger:
        settle("funding_source")

    sells = parsed.value_of("sells_allowed")
    sells_allowed = not (sells and sells.value == "false")

    for ambiguous in parsed.unrecognized:
        question, why = _QUESTIONS["asset_identity"]
        options = AMBIGUOUS_NAMES.get(ambiguous)
        unresolved.append(Unresolved(
            field=f"asset_identity:{ambiguous}",
            question=(f"{question} You wrote '{ambiguous}' — "
                      f"{' or '.join(options)}?" if options
                      else f"{question} You wrote '{ambiguous}'."),
            why_it_matters=why,
        ))

    # Phrases stage 1 could not place. Each becomes a question rather than a
    # default — the rule the whole compiler rests on, applied to whatever a
    # model hands back that the vocabulary has no home for.
    for phrase in parsed.unclear:
        unresolved.append(Unresolved(
            field=f"unclear:{phrase}",
            question=f"What did you mean by '{phrase}'?",
            why_it_matters=(
                "This part of your description did not map onto anything the "
                "compiler can simulate, so it is currently having no effect on "
                "the result."),
        ))

    cadence_rec = parsed.value_of("cadence")
    if not amount_seen:
        question, why = _QUESTIONS["amount"]
        unresolved.append(Unresolved("amount", question, why))
    if not cadence_rec and amount_seen:
        unresolved.append(Unresolved(
            "cadence", "How often does that contribution arrive?",
            "Contribution timing changes the money-weighted return even when the "
            "strategy is identical.",
        ))

    flows = FlowSchedule(
        cadence=cadence_rec.value if cadence_rec else "once",
        amount=float(amount_seen.value) if amount_seen else 0.0,
        day_rule=day_rule or "first_session_of_period",
    )
    if flows.amount <= 0 and flows.starting_capital <= 0:
        question, why = _QUESTIONS["starting_capital"]
        unresolved.append(Unresolved("starting_capital", question, why))

    if benchmark_rule is None:
        question, why = _QUESTIONS["benchmark_set"]
        unresolved.append(Unresolved("benchmark_set", question, why))

    event_program: List[Dict[str, Any]] = []
    if trigger:
        event_program = [
            {"observe": "signal_series"},
            {"condition": "below_moving_average", "semantics": trigger},
            {"action": "buy_basket"},
        ]

    scenario = ScenarioSpecification(
        name=name, version=version, objective=objective,
        event_program=event_program,
        flow_schedule=flows,
        allocation_rule=AllocationRule(
            assets=parsed.assets,
            weighting=weighting or "equal_weight_at_purchase",
        ),
        holdings_policy=HoldingsPolicy(
            dividend_policy=(settled_dividends or "reinvested"),
            sells_allowed=sells_allowed,
            rebalancing_allowed=sells_allowed,
        ),
        benchmark_set=(BenchmarkSet(generated_by_rule=benchmark_rule)
                       if benchmark_rule else None),
        intent_ref=intent_id,
        pending_template=parsed.template_hint,
    )

    # Stage 4 runs against the *compiled* form, not the prose. A check that only
    # reads the input trusts the parser to have been right about it.
    contradictions = [
        Contradiction(between=("holdings_policy", "allocation_rule"), detail=conflict)
        for conflict in scenario.self_conflicts()
    ]

    scenario = ScenarioSpecification(
        **{**scenario.__dict__,
           "provenance": Provenance(stated=tuple(stated), inferred=tuple(inferred),
                                    contradictions=tuple(contradictions),
                                    unresolved=tuple(unresolved))},
    )

    status = ("BLOCKED" if contradictions else
              "NEEDS_INPUT" if unresolved or inferred else "READY")

    # An *offer*, not an open question. It was appended to `unresolved` after
    # the scenario provenance was built, so the confirmation screen listed it
    # under "we still need" while the Save button stayed enabled — the screen
    # and the button disagreed about whether anything was outstanding.
    #
    # Kept non-blocking, because the generic compile is a valid plan and the
    # template is a better one the user chooses. But it no longer sits in the
    # list of things that block, because that list is the one a reader trusts.
    template_offer = None
    if parsed.template_hint:
        template_offer = (Unresolved(
            field=f"template:{parsed.template_hint}",
            question=(
                f"This looks like a {parsed.template_hint.replace('-', ' ')} "
                "situation. Filling in the template asks for the specifics."
            ),
            why_it_matters=(
                "The template encodes vesting, withholding and blackout rules "
                "with citations. Guessing them from your description would "
                "replace checkable rules with a paraphrase of them."
            ),
        ))
        status = "NEEDS_INPUT"

    return CompilerResult(
        intent_id=intent_id, status=status, stated=tuple(stated),
        template_hint=parsed.template_hint, template_offer=template_offer,
        inferred=tuple(inferred), contradictions=tuple(contradictions),
        unresolved=tuple(unresolved), defaults_ref=defaults.artifact_id,
        scenario=scenario,
        verification=tuple(f"self-conflict: {c}" for c in scenario.self_conflicts()),
    )
