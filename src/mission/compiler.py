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
from .representation import representation_gaps
from . import asset_identity, time_window, vocabulary
from .funding import ExecutionTiming
from .spec import AssetResolution, Contradiction, FlowSchedule, Inference, Objective, Provenance, OpenQuestion


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

    observed: Sequence[str] = ()
    """Instruments the sentence *watches*, as distinct from those it buys.

    Two roles, and they can be the same instrument or different ones:

        buy $1,000 of SPY whenever it crosses below   SPY is both
        buy VOO whenever SPY crosses below            VOO held, SPY watched
        notify me whenever SPY crosses below          SPY watched, nothing held

    Derived here and consumed by `_funding_policy`, which previously took the
    watched series to be whichever instrument the plan held — so a plan buying
    VOO on an SPY signal evaluated the condition on VOO, and the two do not
    cross their averages on the same days.
    """

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
            "observed": list(self.observed),
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

#: An event verb applied to the level. "Crosses below" names a transition and
#: fires once per drawdown.
_CROSSING_LANGUAGE = re.compile(
    r"\bcross(?:es|ed|ing)?\b[^.]{0,24}?\b(?:below|above|under|over)\b"
    r"|\b(?:only )?on the day\b[^.]*\bcross(?:es|ed|ing)?\b",
    re.IGNORECASE)

#: State language. "Is below", "stays below", "every day it is below" name a
#: condition that holds, and fire on each session it holds.
#:
#: `below|above` and deliberately not `under|over`: "trades under" is left
#: unrecognised so it is asked about, which is the existing behaviour and the
#: right one — it reads as either to a person.
_PERSISTENT_LANGUAGE = re.compile(
    r"\bwhile\b[^.]*\b(?:below|above)\b"
    r"|\b(?:stays?|staying|stayed|remains?|remaining|remained|sits?|sitting)\b"
    r"[^.]{0,24}?\b(?:below|above)\b"
    r"|\b(?:every|each|any)\s+(?:day|session)\b[^.]*\b(?:below|above)\b"
    r"|\b(?:is|are|was|were|trades?|trading|closes?|closing)\b"
    r"[^.]{0,16}?\b(?:below|above)\b",
    re.IGNORECASE)


#: The closed set this field may take, declared once.
#:
#: `parse_model.VOCABULARY` is derived from `_RULES`, so lifting
#: `trigger_semantics` out of that table silently removed both values from what
#: the model is permitted to propose — and the model layer exists precisely to
#: recognise phrasings the regexes miss. Six tests caught it. Declared here and
#: imported there, so the resolver and the vocabulary cannot disagree.
TRIGGER_SEMANTICS_VALUES = ("crossing_event", "persistent_condition")


def trigger_semantics_ambiguous(text: str) -> bool:
    """Whether the sentence states both readings at once.

    `trigger_semantics` returns None for two different reasons and the caller
    cannot tell them apart: nothing matched, or *both* matched. Those mean
    opposite things. Nothing matched is an absence the model should fill —
    that is what the model layer is for. Both matched is a determination that
    the sentence is ambiguous, and filling it is overriding a decision.

    Left conflated, "whenever it crosses below and stays below" produced no
    deterministic recognition, the model proposed `crossing_event` unopposed,
    `merge` accepted it as new information, and a sentence the compiler had
    judged ambiguous executed on one reader's opinion with no question asked.
    Caught by a production check written to prove the opposite.

    The same absent-versus-empty shape as `provenance@1`: a missing value and
    a value known to be nothing look identical to code that tests only for
    presence.
    """
    return bool(_CROSSING_LANGUAGE.search(text)
                and _PERSISTENT_LANGUAGE.search(text))


def trigger_semantics(text: str) -> Optional[str]:
    """Crossing, persistent, or neither — by precedence, not by list order.

    These are two different rules with materially different results: "every
    time it crosses below" fires once per drawdown, "every day it is below"
    fires on each of them, and over five years that is not a rounding
    difference.

    **An explicit event verb may never be overwritten by a broader
    persistent-condition pattern.** That is the invariant, and it is expressed
    here as precedence rather than as a narrower regex, because a regex
    tightened against one phrase leaves the same hole open for the next
    overlapping matcher someone adds.

    The defect this replaces: the persistent pattern was
    `\\bwhenever\\b[^.]*\\b(?:is |trades |closes )?(?:below|above)\\b` with the
    qualifier optional, so it matched "whenever … below" whatever verb sat
    between them. "I buy $1,000 of SPY whenever it crosses below its 200-day
    moving average" resolved to the persistent reading, silently, and nothing
    was asked — while "when it crosses below" resolved to crossing. One word
    changed a financial rule, and the word the user wrote to say which rule
    they meant was the one discarded. A browser agent found it by reading the
    page back: the plan was rendered as "buys on every day the condition
    holds" for a sentence that says *crosses*.

    Returns None when both readings are present or neither is, so the caller
    raises `trigger_semantics` as a question. Silence is the correct output
    for an ambiguous sentence; a default here is a guess about money.
    """
    crossing = _CROSSING_LANGUAGE.search(text)
    persistent = _PERSISTENT_LANGUAGE.search(text)
    if crossing and persistent:
        return None         # "crosses below and stays below" — genuinely both
    crossing_value, persistent_value = TRIGGER_SEMANTICS_VALUES
    if crossing:
        return crossing_value
    if persistent:
        return persistent_value
    return None


_RULES: Sequence[Tuple[str, str, str]] = (
    # (field, canonical value, pattern)
    #
    # `trigger_semantics` is deliberately absent: it is resolved by
    # `trigger_semantics()` above, which applies precedence between two
    # overlapping vocabularies. A flat first-match-wins table cannot express
    # "an explicit crossing verb outranks a broader state pattern", and
    # expressing it by ordering alone is what failed — the crossing entry was
    # already first, and lost because its pattern did not match at all.

    # When the order fills, when the user says so.
    #
    # There was no rule for this field at all, so "I buy $1,000 of SPY *at the
    # same day's close* whenever it crosses below" was never recognised: the
    # default `next_session_open` was applied and recorded as an *inference*,
    # and the page offered it back as the system's own assumption to confirm.
    # A value the user stated, overwritten, and relabelled as our choice.
    #
    # `settle()` was already correct — stated wins — and never saw a stated
    # value to prefer. `SUPPORTED_TIMING` was already correct too, and refuses
    # same-session close because acting on the close that produced the signal
    # reads one bar into the future. It could not fire either, because the
    # policy always carried the default. Two correct mechanisms, and the input
    # class that needed them never existed.
    ("execution_timing", "same_session_close",
     r"\b(?:at|on)\s+(?:the\s+)?same[- ](?:day|session)'?s?\s+close\b"
     r"|\bsame[- ]day\s+close\b|\bthat\s+(?:day|session)'?s?\s+close\b"
     r"|\bon\s+the\s+close\s+(?:of\s+)?(?:that|the\s+same)\s+(?:day|session)\b"),
    ("execution_timing", "next_session_open",
     r"\bnext\s+(?:session|day|morning)'?s?\s+open\b|\bnext\s+open\b"
     r"|\bopen\s+of\s+the\s+next\s+(?:session|day)\b"),

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
     r"\bfirst trading (?:day|session)\b|\bfirst market day\b|\bfirst session\b"),

    # The other half of a field that only had one. `_flows_from` has always
    # honoured `last_session_of_period` — it takes the group maximum instead of
    # the minimum — and nothing could ever set it, so "$2,000 *last session*
    # every month" and "$2,000 *first session* every month" compiled to the
    # same plan and returned the same figure to the cent. Two descriptions a
    # user would reasonably expect to differ, answered with one number, and
    # neither reading flagged.
    #
    # Deliberately narrow: only phrasings that name a *session*. "month end"
    # and "quarter end" are not here, because "rebalance quarter end" is a
    # sentence about rebalancing, and reading a rebalancing clause as a
    # contribution setting is precisely the defect that turned a single
    # $100,000 allocation into $6,100,000 of monthly contributions. That
    # wording needs the same context guard `cadence` has, and adding it to a
    # flat first-match table without one would reintroduce the known failure.
    ("contribution_day_rule", "last_session_of_period",
     r"\blast trading (?:day|session)\b|\blast market day\b|\blast session\b"
     r"|\bfinal (?:trading )?session\b"),

    # Account type. `tax_treatment` has always been on the scenario and in the
    # content hash, and nothing ever set it — so every plan compiled from prose
    # was NONE_APPLIED and a Roth compared as identical to a taxable account,
    # which is this project's own founding example of a defect.
    #
    # Ordered most specific first: "Roth 401(k)" must not be read as "401(k)",
    # and "Roth IRA" must not be read as "IRA".
    ("account_type", "ROTH_401K", r"\broth 401\s?\(?k\)?\b|\broth401k\b"),
    ("account_type", "ROTH", r"\broth\b"),
    ("account_type", "TRADITIONAL_401K", r"\b401\s?\(?k\)?\b|\b401k\b"),
    ("account_type", "TRADITIONAL_IRA", r"\btraditional ira\b|\btraditional account\b|\bdeductible ira\b"),
    ("account_type", "TAXABLE", r"\btaxable\b|\bbrokerage account\b|\btaxable brokerage\b"),

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
    ("annual", r"\b(?:every|each) year\b|\bannually\b|\bonce a year\b|\byearly\b"
               r"|\b(?:a|per) year\b"),
    ("payroll", r"\bevery pay ?day\b|\beach pay ?day\b|\bout of (?:each|every|my) pay ?che(?:ck|que)\b|\bwith each pay ?che(?:ck|que)\b|\bper pay period\b"),
    ("daily", r"\b(?:every|each) day\b|\bdaily\b"),
    ("once", r"\blump sum\b|\ball at once\b|\bone ?-?off\b"),
)

#: A frequency belonging to rebalancing, not to contributions.
#:
#: "Allocate $100,000 across VTI, BND and GLD by inverse volatility,
#: rebalanced monthly, past 5 years" read `monthly` as a contribution cadence,
#: so a single $100,000 allocation was executed as $100,000 *every month* —
#: $6,100,000 contributed against $100,000 stated, and coverage reported
#: everything declared as executed.
#:
#: The comment on `annual` above records the same defect one word over: a bare
#: adjective matched "annual rebalance" in a benchmark clause. That was fixed
#: by deleting the bare form, which works for an adjective and not for
#: "monthly", a word people genuinely use for contributions. So the context is
#: read instead of the vocabulary being narrowed.
#:
#: How often a portfolio is rebalanced and how often money arrives are
#: different dimensions, and one may not be read out of the other's words.
_REBALANCING_NEARBY = re.compile(r"\brebalanc\w*\b|\brebalance[ds]?\b",
                                 re.IGNORECASE)

#: How far back to look. Long enough for "rebalanced monthly" and "rebalance
#: the portfolio quarterly", short enough that a rebalancing clause earlier in
#: a different sentence does not silence a real contribution cadence.
_REBALANCING_REACH = 34


def _belongs_to_rebalancing(text: str, start: int) -> bool:
    """Whether the frequency word at `start` is describing a rebalance."""
    window = text[max(0, start - _REBALANCING_REACH):start]
    return bool(_REBALANCING_NEARBY.search(window))


def cadence_span_is_rebalancing(text: str, span: str) -> bool:
    """Whether a proposed cadence quotes a rebalancing phrase as its evidence.

    Checked against the model's own span. Having stopped the deterministic
    reader taking "rebalanced monthly" as a contribution cadence, the model
    proposed exactly that — `{"field": "cadence", "value": "monthly", "span":
    "rebalanced monthly"}` — and `merge` accepted it, because a reader that has
    *declined* to read a field looks identical to one that simply did not see
    it. So the plan contributed $100,000 every month again, from the other
    reader.

    Third instance of the shape: silence as a verdict, read as silence as a
    gap. The span makes this one cheap to answer — the model is required to
    quote the words it relied on, so the same context rule can be applied to
    the quotation.
    """
    if not text or not span:
        return False
    at = text.lower().find(span.strip().lower())
    if at < 0:
        return False
    # The frequency word inside the span, not the span's own start: "rebalanced
    # monthly" begins with the disqualifying word, so measuring from the start
    # would look backwards past it.
    inner = re.search(r"\b(?:month|week|quarter|year|day)\w*\b|\bmonthly\b"
                      r"|\bweekly\b|\bquarterly\b|\bannually\b|\bdaily\b",
                      span, re.IGNORECASE)
    offset = at + (inner.start() if inner else len(span))
    return _belongs_to_rebalancing(text, offset)


_AMOUNT = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")

_PERCENT = re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*(?:%|percent\b)",
                      re.IGNORECASE)

#: The same allocation written as a ratio. "60/40" and "60% / 40%" are one
#: semantic object in two notations, and the catalogue uses the second — so
#: the percentage-only reading left "holding 60/40" executing 50/50 with a
#: figure published, which is the defect it was meant to close.
_RATIO_WEIGHTS = re.compile(r"\b(\d{1,3}(?:\s*/\s*\d{1,3}){1,3})\b")

#: How far the stated percentages may be from 100 and still read as an
#: allocation. Two thresholds in a sentence — "falls more than 20%" — do not
#: sum to a portfolio, and that is what keeps this from firing on them.
_ALLOCATION_TOLERANCE = 1.0


def stated_weights(text: str):
    """Per-asset weights the user wrote, or empty.

    "I hold 60% VTI and 40% BND" was compiled to `equal_weight_at_purchase`
    and executed 50/50, with a figure published. `AllocationRule` has no
    per-asset weights field at all, so the numbers had nowhere to go — and
    over five years the difference between a 60/40 and a 50/50 equity/bond
    split is not a rounding error.

    Recognised as an allocation only when the percentages *sum to about 100*.
    A percentage in a sentence is usually a threshold — "after the market
    falls more than 20%" — and thresholds do not add up to a portfolio.
    """
    found = [float(one) for one in _PERCENT.findall(text or "")]
    if len(found) < 2:
        # Ratio notation, when percentages were not used. Tried second so a
        # sentence writing both does not count its weights twice.
        for match in _RATIO_WEIGHTS.finditer(text or ""):
            parts = [float(one) for one in re.split(r"\s*/\s*", match.group(1))]
            if len(parts) >= 2 and abs(sum(parts) - 100.0) <= _ALLOCATION_TOLERANCE:
                return tuple(parts)
        return ()
    if abs(sum(found) - 100.0) > _ALLOCATION_TOLERANCE:
        return ()
    return tuple(found)


def weights_are_equal(weights, asset_count: Optional[int] = None) -> bool:
    """Whether stated weights say exactly what `equal_weight_at_purchase` says.

    "50% VTI and 50% BND" is not an unsupported allocation; it is the
    supported one, written as numbers. Blocking it because numbers were used
    would refuse a plan the engine executes correctly — the over-reach this
    area has walked back twice.

    **Zeros are a claim about a holding, not padding.** "100/0" naming one
    instrument means all of it, which is what equal-weighting one holding
    does. "100/0" naming two means all of the first and none of the second,
    which is not. So the non-zero weights must be equal *and* there must be
    one of them per asset; without the asset count this cannot be decided, and
    the safe answer is that it is not equivalent.
    """
    if not weights:
        return False
    positive = [one for one in weights if one > 0]
    if not positive:
        return False
    if max(positive) - min(positive) > _ALLOCATION_TOLERANCE:
        return False
    if asset_count is None:
        return len(positive) == len(weights)
    return len(positive) == asset_count

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
#: The averaging window, in sessions, as the description states it. Extracted
#: rather than defaulted: "200-day" and "50-day" are different rules, and a
#: compiler that assumed 200 would answer a question about a 50-day average
#: with a number that looks right. A moving average named with no window
#: becomes an unresolved field, like every other unstated choice.
_AVERAGE_WINDOW = re.compile(
    r"\b([0-9]{1,4})[-\s]?(?:day|session)s?\b[^.]{0,24}?\baverage\b"
    r"|\b([0-9]{1,4})[-\s]?(?:DMA|SMA|EMA)\b",
    re.IGNORECASE)
_TICKER = re.compile(r"\b([A-Z]{1,5})\b")


def moving_average_window(text: str) -> Optional[int]:
    """The averaging window the description states, or None."""
    found = _AVERAGE_WINDOW.search(text)
    if not found:
        return None
    digits = found.group(1) or found.group(2)
    window = int(digits)
    return window if 2 <= window <= 2000 else None

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

    # Before the flat table, because this one is decided by precedence between
    # two vocabularies rather than by whichever pattern is listed first.
    # Equal percentages *are* the supported weighting, written as numbers.
    # Recorded as a recognition so it is STATED rather than inferred: the user
    # said it, and describing their words as the compiler's own assumption is
    # the authority inversion this codebase keeps finding.
    weights = stated_weights(text)
    if weights and weights_are_equal(weights):
        found_span = _PERCENT.search(text)
        recognitions.append(Recognition(
            field="weighting", value="equal_weight_at_purchase",
            span=found_span.group(0) if found_span else ""))
        claimed.add("weighting")

    semantics = trigger_semantics(text)
    if semantics is not None:
        source = (_CROSSING_LANGUAGE if semantics == "crossing_event"
                  else _PERSISTENT_LANGUAGE).search(text)
        recognitions.append(Recognition(
            field="trigger_semantics", value=semantics,
            span=source.group(0).strip() if source else ""))
        claimed.add("trigger_semantics")

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
        if "cadence" in claimed:
            break
        # Every occurrence, not just the first: a plan may rebalance monthly
        # and contribute monthly in one sentence, and taking the first match
        # would let the rebalancing clause consume the contribution's word.
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _belongs_to_rebalancing(text, match.start()):
                continue
            recognitions.append(
                Recognition("cadence", cadence, match.group(0).strip()))
            claimed.add("cadence")
            break

    amount = _AMOUNT.search(text)
    if amount:
        recognitions.append(
            Recognition("amount", amount.group(1).replace(",", ""), amount.group(0)))

    lowered = text.lower()
    unrecognized = [name for name in AMBIGUOUS_NAMES if name in lowered]

    # `SPY` is reserved because it is usually the *reference* in a trend rule —
    # "whenever SPY is below its 200 day average" names a signal, not a holding.
    # But "I buy $500 of SPY every week" names a holding, and the reserved list
    # made that compile to no assets at all. The model read it correctly and the
    # rules did not; the fix belongs in the rules.
    reserved = {"DMA", "EMA", "SMA", "RSU", "ESPP", "IRA"}

    # An instrument can be watched *and* bought, and the reservation used to
    # deny it.
    #
    # "whenever SPY is below its 200 day average" names a condition, not a
    # holding — correct, and implemented by reserving SPY away from the asset
    # list whenever it appeared near signal language. In the sentence that does
    # both:
    #
    #     I buy $1,000 of SPY whenever it crosses below its 200-day average
    #
    # the reservation consumed it entirely. The plan held nothing, no funding
    # policy could be built, and the most natural phrasing of the strategy was
    # the one shape that could not run — while the message said the rule had
    # not been executed.
    #
    # The distinction is not proximity to signal language. It is whether the
    # sentence gives an action that acquires the instrument:
    #
    #     mentioned only as the thing observed  -> signal subject only
    #     also the object of a buy              -> subject and holding
    observed = _observed_in_signal(text)
    acquired = _acquired_instruments(text)
    reserved = reserved | (observed - acquired)
    assets = [t for t in _TICKER.findall(text) if t not in reserved and len(t) >= 2]

    hint = next((name for name, pattern in _TEMPLATE_HINTS
                 if pattern.search(text)), None)

    return ParsedUtterance(text=text, recognitions=tuple(recognitions),
                           assets=tuple(dict.fromkeys(assets)),
                           observed=tuple(sorted(observed)),
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
    "account_type": (
        "Which account is this in — taxable, traditional IRA or 401(k), Roth "
        "IRA or Roth 401(k)?",
        "Tax treatment changes the result more than most rules do: the same "
        "contributions compound tax-free in a Roth, tax-deferred in a "
        "traditional account, and after tax on dividends and gains in a taxable "
        "one. Guessing would let a Roth and a taxable plan compare as "
        "identical, which is the defect this whole system exists to prevent.",
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
    unresolved: Sequence[OpenQuestion]
    defaults_ref: str
    scenario: Optional[ScenarioSpecification]
    verification: Sequence[str] = ()
    template_hint: Optional[str] = None
    template_offer: Optional[OpenQuestion] = None
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


def _as_amount(raw: Optional[str]) -> Optional[float]:
    """An answered amount, or None. A malformed answer is not an amount, and
    treating it as zero would silently produce a plan with no contributions."""
    if raw is None:
        return None
    try:
        return float(str(raw).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


#: Words a model uses when it is describing a field the compiler already owns.
#:
#: Conservative on purpose: a phrase is only suppressed when the field it
#: names is *also* being asked or confirmed on the same page, so nothing is
#: lost by a match — the user answers the structured control instead. A missed
#: match costs an extra acknowledgement, which is the safe direction.
#: Ticker-shaped tokens. Uppercase only: a lowercase word is prose.
_TOKEN = re.compile(r"\b([A-Z][A-Z0-9.\-]{1,4})\b")

_SIGNAL_PHRASE = re.compile(
    r"\b(?:cross(?:es|ing|ed)?|below|above|moving average|DMA|SMA|EMA)\b",
    re.IGNORECASE)

_ACQUIRING_VERB = re.compile(
    r"\b(?:buy|buys|buying|bought|purchase[sd]?|acquire[sd]?|invest(?:ing|ed)?|"
    r"put(?:ting)?|add(?:ing)?|contribut(?:e|es|ing))\b",
    re.IGNORECASE)

#: How far a role marker reaches. A sentence puts the verb and its object, or
#: the subject and its condition, close together; sixty characters spans a
#: clause and not the sentence.
_ROLE_REACH = 60


def _sentence_bounds(text: str, position: int) -> tuple:
    """The sentence containing `position`.

    A role never spans a full stop. Rendering a plan back to English produces
    two sentences —

        I put $500 into VTI, every month, buying equal dollars at each
        purchase. Whenever SPY is below its 200 day average I buy more of VTI.

    — and a reach measured in characters let "buying" at the end of the first
    claim SPY at the start of the second. The plan came back holding SPY, and
    its rule hash drifted on every round trip.
    """
    start = max((text.rfind(mark, 0, position) for mark in ".!?"), default=-1)
    ends = [e for e in (text.find(mark, position) for mark in ".!?") if e != -1]
    return start + 1, (min(ends) if ends else len(text))


def _nearest_token_before(text: str, position: int) -> Optional[str]:
    start, _ = _sentence_bounds(text, position)
    candidates = [m for m in _TOKEN.finditer(text, start, position)
                  if position - m.end() <= _ROLE_REACH]
    return candidates[-1].group(1) if candidates else None


def _nearest_token_after(text: str, position: int) -> Optional[str]:
    _, end = _sentence_bounds(text, position)
    for match in _TOKEN.finditer(text, position, end):
        if match.start() - position <= _ROLE_REACH:
            return match.group(1)
    return None


def _acquired_instruments(text: str) -> set:
    """Tickers the sentence says are bought.

    Anchored to the verb and taking the nearest token after it. A proximity
    test over the whole clause read "Buy VOO whenever SPY crosses below" as
    acquiring both, because SPY sits a few words from "Buy".
    """
    found = set()
    for verb in _ACQUIRING_VERB.finditer(text):
        token = _nearest_token_after(text, verb.end())
        if token and len(token) >= 2 and token.isalpha():
            found.add(token)
    return found


def _observed_in_signal(text: str) -> set:
    """Tickers the sentence says are watched.

    Anchored to the signal phrase and taking the nearest token *before* it —
    the subject of "crosses below" is what precedes it. Scanning outward from
    the ticker instead made every instrument in the clause observed, so
    "Buy VOO whenever SPY crosses below" watched VOO.
    """
    found = set()
    for phrase in _SIGNAL_PHRASE.finditer(text):
        token = _nearest_token_before(text, phrase.start())
        if token and len(token) >= 2 and token.isalpha():
            found.add(token)
    return found


_CONCEPT_MARKERS = {
    "moving_average_kind": ("simple or exponential", "exponential moving average",
                            "type of moving average", "simple vs exponential"),
    "trigger_semantics": ("one-time crossing", "persistent condition",
                          "buying repeatedly", "every time the condition",
                          "one time or repeated"),
    "cadence": ("cadence", "frequency of the", "how often"),
    "funding_source": ("new contribution vs", "source of the",
                       "existing cash"),
    "account_type": ("account type",),
    "amount": ("amount to invest", "how much is invested"),
    "starting_capital": ("starting capital", "initial balance"),
    "dividends": ("dividend treatment", "dividends are reinvested"),
    "execution_timing": ("when orders execute", "execution timing"),
    # Concepts the compiler settles *without* asking, which the parser still
    # reports uncertainty about. "five year period mentioned" and "no
    # recurring cadence specified" both describe decisions already made: the
    # window was detected, and cadence does not apply to event funding.
    "time_window": ("year period", "years period", "period mentioned",
                    "lookback", "backtest period", "time period",
                    "evaluation period", "timeframe", "date range"),
    "cadence_inapplicable": ("recurring cadence", "no cadence", "cadence "
                             "specified", "recurring schedule", "no schedule",
                             "frequency specified", "no frequency"),
}


def _covered_by(phrase: str, raised: set,
                settled: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The structured fact this phrase is describing, if the compiler has it.

    Two ways a phrase stops being unclear, and only the first was implemented.

    **The field is still being asked.** The phrase describes a question already
    on the page, so filing it separately would ask twice.

    **The compiler already has the value.** The parser may report uncertainty
    about something the deterministic stages went on to resolve — and it does:
    a description stating "$1,000" and "200-day" produced

        unclear: 1,000 purchase amount per trigger
        unclear: 200-day period length for moving average

    while the compiler had recognised the amount and extracted the window and
    used both. The parser's uncertainty outlived the compiler's certainty, and
    the user was asked to dismiss facts their plan had actually modelled.

    The parser says a phrase *may* be unclear. Whether it is still unclear —
    after recognitions, inferences and amendments — is the compiler's call.

    The numeric test is the general one. Markers are a hand-kept list of the
    ways a model might phrase a concept, and the two phrases above matched none
    of them; a figure the compiler holds is a fact, however the phrase around
    it is worded.
    """
    import re as _re

    lowered = phrase.lower()
    for field, markers in _CONCEPT_MARKERS.items():
        if field in raised and any(marker in lowered for marker in markers):
            return field

    numbers = {n.replace(",", "") for n in _re.findall(r"[\d,]*\d", lowered)}
    if numbers and settled:
        for field, value in settled.items():
            if value is None:
                continue
            held = str(value).replace(",", "")
            held = held[:-2] if held.endswith(".0") else held
            if held and held in numbers:
                return field
    return None



def canonical_key(prefix: str, phrase: str) -> str:
    """A field id that survives the model rewording its own explanation.

    The clarification loop could not converge. `unclear` entries carry the
    model's commentary — "SP500 ETF (company/product name, not a literal
    ticker symbol)" — and the field id was built from the whole string, so six
    rounds produced six ids for one question:

        asset_identity:SP500 ETF (company/product name, not a literal ticker…)
        asset_identity:SP500 ETF (ticker symbol not specified)
        asset_identity:SP500 ETF (fund name given, not a literal ticker symbol)

    An answer to the first did not match the second, so the question returned,
    reworded, forever. Five of nine recorded journeys never settled.

    **The invariant is semantic, not punctuational:**

        The persistent key is derived from the stable observed subject, never
        from explanatory model prose.

    Dropping parentheses is how that is achieved for the wording seen; it is
    not the rule. Prose that arrives as a dash clause, a trailing comma or a
    second sentence is the same defect in different punctuation, and the
    normalisation below is expected to grow to cover it. What must not change
    is where the key comes from: the subject the user named, and nothing the
    model said about it.

    The model may report the observed phrase, why it is ambiguous, and the
    candidates. The commentary stays in the question text, where changing
    wording costs nothing.

    A digest is the fallback rather than the rule. `unclear:#<hash>` is
    unreadable in a form field and in a log, and readability is worth having
    wherever the phrase itself is safe to keep.
    """
    import hashlib
    import re as _re

    # Cut at the first *explanatory boundary*, whatever punctuation carries it.
    #
    # The first version stripped parentheses and dash clauses, because those
    # were the forms the live model had produced. Three other forms of the same
    # sentence walked straight through it:
    #
    #     SP500 ETF. A ticker symbol was not provided.
    #     SP500 ETF: fund name rather than a ticker
    #     SP500 ETF, which is a product name not a ticker
    #
    # each yielding a different key for one subject. Matching the punctuation
    # seen is implementing the example; the invariant is that the key stops
    # where the subject stops.
    base = _re.sub(r"\s*[\(\[].*?(?:[\)\]]|$)", " ", phrase or "")
    base = _re.split(r"\s+[-—–]\s+|[.:;]\s|[.:;]$", base)[0]
    # A comma only ends the subject when a clause follows it. "SPDR S&P 500
    # ETF Trust, Inc" is one name; "SP500 ETF, which is a product name" is a
    # subject and a remark.
    base = _re.split(
        r",\s+(?:which|that|this|it|a|an|the|not|no|but|rather|meaning|"
        r"referring|referenced|i\.e\.|e\.g\.)\b",
        base, flags=_re.IGNORECASE)[0]
    slug = _re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if not slug:
        slug = "x" + hashlib.sha256((phrase or "").encode()).hexdigest()[:10]
    return f"{prefix}:{slug[:60]}"


def _funding_policy(*, trigger, parsed, amount, cadence, day_rule,
                    assets, window=None, priceable=(), execution_timing=None):
    """The single authority on how money arrives.

    Built here and nowhere else, because two builders of a funding policy would
    be two answers to the question the sum type exists to make unambiguous.

    Returns `None` when the amount is still unresolved: a policy with no amount
    would state that money arrives without saying how much, and an unresolved
    field must stay unresolved rather than becoming a zero.
    """
    from decimal import Decimal

    from .funding import (
        Estimator,
        EventTriggered,
        Scheduled,
        Trigger,
    )
    from .signals import SignalKind

    if amount is None:
        return None

    if trigger:
        average = parsed.value_of("moving_average_kind")
        # The first asset this deployment can actually price. "SP500 ETF"
        # compiles to ('ETF', 'SPY') — the phrase's own token and the
        # instrument it resolved to — and evaluating a moving average on a
        # ticker with no price history would refuse a plan that is perfectly
        # runnable on the instrument it actually holds.
        # The watched series, when the sentence names one this deployment can
        # price. Falling straight through to the held instrument made a plan
        # buying VOO on an SPY signal evaluate the condition on VOO — a
        # different rule, silently.
        priceable_set = set(priceable)
        watched = [a for a in (parsed.observed or ()) if a in priceable_set]
        priceable_assets = [a for a in assets if a in priceable_set]
        subject = next(iter(watched), "") or next(iter(priceable_assets), "")
        if not subject or window is None:
            # No policy rather than a guessed one. An unstated window is an
            # unresolved field, and the plan is blocked on it like any other.
            return None
        return EventTriggered(
            trigger=Trigger(
                subject=subject,
                window=window,
                estimator=Estimator(average.value if average else "simple"),
                # The user's answer, honoured. "Every time it crosses below"
                # fires once per drawdown and "every day it is below" fires on
                # each of them; over five years that is not a rounding
                # difference, and the compiler asks precisely so this choice is
                # theirs rather than the engine's.
                kind=(SignalKind.CROSSED_BELOW_MOVING_AVERAGE
                      if trigger == "crossing_event"
                      else SignalKind.BELOW_MOVING_AVERAGE)),
            amount=Decimal(str(amount)),
            # Consumed, not merely settled. The field was recognised, resolved
            # by `settle` and then dropped here, so the policy always carried
            # the default and `SUPPORTED_TIMING` — which exists to refuse
            # same-session close — never saw a value to refuse. Recognised,
            # settled, and no consumption site: the same shape as an amendment
            # that reaches nothing.
            **({"execution_timing": ExecutionTiming(execution_timing)}
               if execution_timing else {}))

    return Scheduled(cadence=cadence or "once", amount=Decimal(str(amount)),
                     day_rule=day_rule or "first_session_of_period")


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
    amendments: Sequence["ScenarioAmendment"] = (),
    exclusions: Sequence["ScenarioExclusion"] = (),
    priceable: Sequence[str] = (),
) -> CompilerResult:
    """Stages 1–8. Deterministic from the parse onward.

    `amendments` are answers the user gave to questions this compiler asked on
    a previous pass. They are consulted *before* defaults and recorded as
    stated, because that is what they are: the user supplying a value. The
    original text is never edited and no answer becomes an inference — see
    `ScenarioAmendment`.

    `exclusions` are parts of the description the user was told could not be
    represented and chose to proceed without. They drop the corresponding
    question rather than answering it, and the scenario records that its scope
    is narrower than its description.

    `priceable` is what the deployment can actually value. Identity candidates
    are filtered to it, because offering a fund the pilot cannot price would
    replace one dead end with a politer one.

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
    unresolved: List[OpenQuestion] = []
    _answers = {one.question_id: one for one in amendments}
    _excluded_items = {one.item for one in exclusions}

    def answered(field_name: str) -> Optional[str]:
        """A user's answer to a question about `field_name`, recorded as stated.

        Every `unresolved.append` site consults this. Wiring it only into
        `settle` would leave the questions raised elsewhere — cadence, amount,
        starting capital — unanswerable through the product, which is the
        defect that blocked the RSU and Roth scenarios.
        """
        one = _answers.get(field_name)
        if one is None:
            return None
        # An answer outside the field's vocabulary is not an answer.
        #
        # Nothing checked this: `cadence=banana` removed the question and
        # recorded "cadence: banana (answered)" as a stated fact, so a saved
        # plan could carry a cadence the renderer has no word for and the
        # engine no schedule for. Treated as unanswered, the question survives
        # and the user is asked again — which is the honest outcome for a
        # value the system cannot use.
        if not vocabulary.accepts(field_name, one.answer):
            return None
        stated.append(f"{field_name}: {one.answer} (answered)")
        return one.answer

    def settle(field_name: str, default_field: Optional[str] = None) -> Optional[str]:
        """Stated wins; then a user's answer; then a *versioned* default;
        otherwise a question.

        An answer outranks a default and is recorded as stated, not inferred.
        Ordering it after the description is deliberate: if the text already
        says something, that is what the user wrote, and an answer to a
        question the text had already settled is not a question that was asked.
        """
        found = parsed.value_of(field_name)
        if found:
            return found.value
        supplied = answered(field_name)
        if supplied is not None:
            return supplied
        entry = defaults.get(default_field or field_name)
        if entry:
            inferred.append(Inference(field=field_name, value=entry.value,
                                      why=entry.why, confirmed=False))
            return entry.value
        question, why = _QUESTIONS.get(
            field_name, (f"What should {field_name.replace('_', ' ')} be?",
                         "This changes the result."))
        unresolved.append(OpenQuestion(field=field_name, question=question,
                                     why_it_matters=why))
        return None

    # Ask only about what the description implies. A question about a condition
    # nobody mentioned is noise at best, and at worst it blocks a complete plan
    # on an ambiguity that does not exist in it.
    has_signal = bool(_MENTIONS_SIGNAL.search(text))
    execution_timing_value = None
    trigger = settle("trigger_semantics") if has_signal else None
    average_window = moving_average_window(text) if has_signal else None
    if has_signal and _MENTIONS_AVERAGE.search(text):
        settle("moving_average_kind")
        if average_window is None:
            # The answer, consumed. The question was raised, a control was
            # rendered for it, the reply was recorded as an amendment — and
            # nothing read it, so the question came back every round.
            #
            # The same family as the asset-identity defect and a different
            # stage of it: that key was unstable so the answer could not
            # match; this key is stable, the answer matches, and the compiler
            # never asked for it. A settle site is not implied by a stable id.
            chosen = answered("moving_average_window")
            if chosen and str(chosen).strip().isdigit():
                average_window = int(str(chosen).strip())
            else:
                unresolved.append(OpenQuestion(
                    "moving_average_window",
                    "How many sessions does the average cover?",
                    "A 50-session and a 200-session average cross on different "
                    "days, so they are different rules producing different "
                    "purchases. Assuming one would answer a question you did "
                    "not ask."))
    if has_signal:
        execution_timing_value = settle("execution_timing")

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

    account = settle("account_type")
    sells = parsed.value_of("sells_allowed")
    sells_allowed = not (sells and sells.value == "false")

    # Assets the user named by clarification rather than by ticker. They join
    # the parsed assets rather than replacing the description.
    identified: List[str] = []
    for ambiguous in parsed.unrecognized:
        question, why = _QUESTIONS["asset_identity"]
        options = AMBIGUOUS_NAMES.get(ambiguous)
        # An answer settles it. Without this the question was raised, an input
        # was rendered for it, the reply was recorded as an amendment, and the
        # same question came back — the field had no settle site at all.
        key = canonical_key("asset_identity", ambiguous)
        chosen = answered(key)
        if chosen:
            identified.append(chosen)
            continue
        unresolved.append(OpenQuestion(
            field=key,
            question=(f"{question} You wrote '{ambiguous}' — "
                      f"{' or '.join(options)}?" if options
                      else f"{question} You wrote '{ambiguous}'."),
            why_it_matters=why,
        ))

    # Phrases that name an asset. Asked as identity rather than filed under
    # "could not place": "SPX ETF" is an index and a fund request in one
    # breath, and "there is no price history for SPX" answers a question
    # nobody asked. Offering the funds that track it is the useful reply.
    # The temporal instruction, from the description itself. Recognised before
    # the unclear loop so that a phrase describing it is not also filed as
    # unplaceable prose the user can only acknowledge away.
    window = time_window.detect(text)

    _still_unclear = []
    asset_resolutions: List[AssetResolution] = []
    for phrase in parsed.unclear:
        # A phrase that *is* the time window has a home now.
        if window is not None and time_window.detect(phrase) is not None:
            stated.append(f"time window: {window.label} (from the description)")
            continue
        found = asset_identity.identify(phrase, priceable=priceable)
        if not found.candidates:
            _still_unclear.append(phrase)
            continue
        field = canonical_key("asset_identity", phrase)
        chosen = answered(field)
        # Recorded whether or not it has been answered yet. The alternatives a
        # user was shown are part of what happened, and a plan that stores
        # only the outcome cannot say what the choice was between.
        record = AssetResolution(
            observed_phrase=phrase,
            registry_digest=found.registry_digest,
            resolved_concept_id=found.concept_id,
            concept_name=found.reason.split(" is an index")[0]
            if " is an index" in found.reason else "",
            candidates_shown=tuple(one.symbol for one in found.candidates),
            chosen_instrument_id=chosen or "",
            ranking_reasons=tuple(
                f"{one.symbol}: {'; '.join(one.reasons)}"
                for one in found.candidates),
        )
        asset_resolutions.append(record)
        if chosen:
            identified.append(chosen)
            continue
        offered = " or ".join(
            f"{one.symbol} ({one.name})" for one in found.candidates)
        unresolved.append(OpenQuestion(
            field=field,
            question=f"You wrote '{phrase}'. Did you mean {offered}?",
            why_it_matters=(found.reason or "") + (
                " Nothing is rewritten — your description keeps the words you "
                "used, and the plan records which asset you meant."),
        ))


    cadence_rec = parsed.value_of("cadence")
    cadence_value = cadence_rec.value if cadence_rec else answered("cadence")
    amount_value = (float(amount_seen.value) if amount_seen
                    else _as_amount(answered("amount")))
    if amount_value is None:
        question, why = _QUESTIONS["amount"]
        unresolved.append(OpenQuestion("amount", question, why))
    if not cadence_value and amount_value and not has_signal:
        # Not asked when the description mentions a condition at all, rather
        # than asked and discarded. The user answered "once" to this question
        # and it became one contribution for a five-year rule — a question
        # whose answer the plan cannot honour is worse than no question,
        # because the user is entitled to believe their answer did something.
        #
        # Gated on `has_signal` rather than on the trigger being settled: the
        # trigger settles later, so the narrower test asked for a cadence on
        # the first pass and withdrew the question on the second. A question
        # that appears and vanishes is the same defect one round trip later.
        unresolved.append(OpenQuestion(
            "cadence", "How often does that contribution arrive?",
            "Contribution timing changes the money-weighted return even when the "
            "strategy is identical.",
        ))

    funding_source = parsed.value_of("funding_source")

    held = tuple(dict.fromkeys(tuple(parsed.assets) + tuple(identified)))
    funding_policy = _funding_policy(
        trigger=trigger, parsed=parsed, amount=amount_value,
        cadence=cadence_value, day_rule=day_rule,
        assets=held, window=average_window, priceable=priceable,
        execution_timing=execution_timing_value)

    # How money arrives is one question with two policies, and a trigger is one
    # of them.
    #
    # Written as a cadence *and* a rule, a plan reading "buy $1,000 every time
    # SPY crosses below its 200-day average" compiled to `cadence=once` with
    # the rule recorded beside it — one contribution across five years, and a
    # figure that was really buy-and-hold. The schedule an event-funded plan
    # carries states no cadence and no amount, so nothing downstream can read a
    # schedule that was never described; `self_conflicts` refuses a scenario
    # where both are stated.
    # Gated on a policy having been *built*, not merely on a trigger being
    # present. Emptied on the trigger alone, a plan whose subject cannot be
    # priced — or whose window is unstated — lost its schedule and gained no
    # policy, so it declared no money at all and was refused for having none.
    # The honest state for such a plan is the Deployment 1 one: a schedule it
    # keeps, and a rule this build refuses to execute.
    from .funding import EventTriggered as _EventTriggered

    event_funded = isinstance(funding_policy, _EventTriggered)
    if event_funded:
        flows = FlowSchedule(
            cadence="event_triggered", amount=0.0,
            day_rule=day_rule or "first_session_of_period",
            funding_source=funding_source.value if funding_source else "contribution",
        )
    else:
        flows = FlowSchedule(
            cadence=cadence_value or "once",
            amount=amount_value or 0.0,
            day_rule=day_rule or "first_session_of_period",
            funding_source=funding_source.value if funding_source else "contribution",
        )
    if (not event_funded and flows.amount <= 0
            and flows.starting_capital <= 0):
        question, why = _QUESTIONS["starting_capital"]
        unresolved.append(OpenQuestion("starting_capital", question, why))

    if window is not None and not window.supported:
        unresolved.append(OpenQuestion(
            field=f"time_window:{window.kind.value}",
            question=(f"You wrote '{window.observed}'. This build can replay a "
                      f"trailing period — 'the past 5 years' — but not that."),
            why_it_matters=(
                "Reading it as a trailing window would answer a different "
                "question with a number that looks right. The description is "
                "unchanged; say a trailing period instead and the plan runs.")))

    if benchmark_rule is None:
        question, why = _QUESTIONS["benchmark_set"]
        unresolved.append(OpenQuestion("benchmark_set", question, why))

    event_program: List[Dict[str, Any]] = []
    if trigger:
        # The estimator belongs in the program, not only in the confirmation
        # text. A simple and an exponential moving average cross at different
        # times, so they are different rules — and before this they produced an
        # identical content hash. Found by the representation check below, which
        # exists because the same omission had already happened to dividends.
        average = parsed.value_of("moving_average_kind")
        event_program = [
            {"observe": "signal_series",
             "estimator": (average.value if average else "simple")},
            {"condition": "below_moving_average", "semantics": trigger},
            {"action": "buy_basket"},
        ]

    if funding_policy is not None and trigger:
        # Which series the condition is evaluated on, disclosed as an
        # inference rather than assumed.
        #
        # "the S&P 500 crosses below its 200-day average" names an index, and
        # this build prices instruments. The condition is therefore evaluated
        # on the instrument the plan holds — which is a real assumption with a
        # real consequence (an ETF and its index diverge by fees and tracking
        # error), and the user is entitled to see it stated and to reject it.
        inferred.append(Inference(
            field="signal_series",
            value=funding_policy.trigger.subject,
            why=(f"The condition is evaluated on {funding_policy.trigger.subject}, "
                 f"the instrument this plan holds. An index itself is not "
                 f"priceable here, and an index and a fund tracking it do not "
                 f"cross an average on identical days.")))

    scenario = ScenarioSpecification(
        name=name, version=version, objective=objective,
        event_program=event_program,
        flow_schedule=flows,
        allocation_rule=AllocationRule(
            assets=held,
            weighting=weighting or "equal_weight_at_purchase",
        ),
        holdings_policy=HoldingsPolicy(
            dividend_policy=(settled_dividends or "reinvested"),
            sells_allowed=sells_allowed,
            rebalancing_allowed=sells_allowed,
        ),
        benchmark_set=(BenchmarkSet(generated_by_rule=benchmark_rule)
                       if benchmark_rule else None),
        tax_treatment=account or "NONE_APPLIED",
        intent_ref=intent_id,
        pending_template=parsed.template_hint,
        funding=funding_policy,
    )

    # Stage 4 runs against the *compiled* form, not the prose. A check that only
    # reads the input trusts the parser to have been right about it.
    contradictions = [
        Contradiction(between=("holdings_policy", "allocation_rule"), detail=conflict)
        for conflict in scenario.self_conflicts()
    ]


    # Phrases stage 1 could not place. Raised last, once every structured
    # question and inference exists, because most of them are not unplaceable
    # at all — they are the model describing, in prose, a field the compiler
    # already asks about properly.
    #
    # A live pilot page carried "What did you mean by 'account type in which
    # the purchases occur is not specified'?" — offering only "continue
    # without modelling it" — directly beneath the account-type question and
    # its five radio buttons. Same for the moving-average kind, the trigger
    # semantics, the cadence and the funding source. The user was asked to
    # abandon something the page was simultaneously offering to settle, with
    # no way to tell which question would actually be answered.
    #
    # So a phrase is dropped when the field it describes is already on the
    # page. Only then: if nothing covers it, it is genuinely unplaceable and
    # the acknowledgement is the honest control.
    _raised = {one.field for one in unresolved} | {one.field for one in inferred}

    # What the compiler ended up holding, for the numeric test above. Built
    # from the values it will actually run with, not from what was recognised
    # — an amount that was recognised and then overridden by an amendment is
    # settled at the amended value.
    _settled_values = {
        "amount": amount_value,
        "moving_average_window": average_window,
        "starting_capital": getattr(flows, "starting_capital", None),
    }
    # Concepts the compiler has decided without raising a question. A phrase
    # describing one of these is not unplaceable prose — it is the parser
    # remarking on a decision that has already been made, and asking the user
    # to dismiss it would record them proceeding without something the plan
    # did use.
    _settled_concepts = set()
    if window is not None:
        _settled_concepts.add("time_window")
    if event_funded:
        # Cadence is not merely unanswered here; it does not apply. The
        # builder deliberately stops asking it once funding is event-driven,
        # and a phrase noting its absence must not become a blocker.
        _settled_concepts.add("cadence_inapplicable")

    for phrase in _still_unclear:
        covered = _covered_by(phrase, _raised | _settled_concepts,
                              _settled_values)
        if covered:
            stated.append(
                f"{covered}: asked directly rather than as unplaceable prose")
            continue
        unresolved.append(OpenQuestion(
            field=canonical_key("unclear", phrase),
            question=f"What did you mean by '{phrase}'?",
            why_it_matters=(
                "This part of your description did not map onto anything the "
                "compiler can simulate, so it is currently having no effect on "
                "the result."),
        ))

    scenario = ScenarioSpecification(
        **{**scenario.__dict__,
           "provenance": Provenance(stated=tuple(stated), inferred=tuple(inferred),
                                    contradictions=tuple(contradictions),
                                    unresolved=tuple(
                                        one for one in unresolved
                                        if one.field not in _excluded_items),
                                    amended=tuple(amendments),
                                    excluded=tuple(exclusions),
                                    asset_resolutions=tuple(asset_resolutions),
                                    time_window=window)},
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
        template_offer = (OpenQuestion(
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
        verification=tuple(
            [f"self-conflict: {c}" for c in scenario.self_conflicts()]
            + [f"unrepresented: {g}" for g in representation_gaps(parsed, scenario)]
        ),
    )
