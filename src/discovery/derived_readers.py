"""Deterministic readers that derive one narrow semantic from syntax evidence.

    Syntax evidence does not author intent.
    Certified deterministic semantic readers may derive narrowly defined
    intent from syntax evidence.

That distinction is the whole module. Stanza produces a parse; `semantics`
turns the parse into candidates; neither may carry a contract field, because a
parser is evidence and not authority. What may carry a field is a *reader* —
something with a name, a version, and a contract about exactly what it will
claim — and that is what lives here.

**Why this exists rather than the two alternatives.** The live drift lane found
"buy VOO when SPY falls below its 200-day moving average" executing on two
draws of five and asking a question on the other three, because the hosted
reader non-deterministically omits `trigger_semantics` altogether. Two obvious
responses were both wrong:

- *Always ask.* Converges, and turns a supported journey into a follow-up on
  every event-triggered sentence, including the ones whose grammar states the
  answer outright.
- *Let syntax carry the field.* Converges, and makes the parser an authority on
  meaning, which is the thing the fusion policy exists to prevent.

The third way is to name the thing in between. `TriggerSemanticsReader` is a
reader like the hosted one: it makes a claim, under an id, and fusion weighs it
against the other reader's claim by the ordinary rules. No source type wins
automatically — "no privileged reader" means disagreements are not settled by
provenance, not that every material fact needs two independent witnesses before
it may exist at all. Requiring the latter would make stochastic model
participation mandatory for sentences whose semantics are deterministically
clear.

**The restriction is structural.** This reader may author `trigger_semantics`
and nothing else, asserted by `tests/test_derived_readers.py`. Left unbounded
it would grow a field at a time back into `quantify-compiler@2`, which took
months to delete.
"""
from __future__ import annotations

import re

from typing import Optional, Sequence

from .fusion import Proposal

#: Versioned like every other reader. A derived reading whose rules changed
#: under a fixed id would make two runs look comparable when they are not.
TRIGGER_READER_ID = "quantify-trigger-semantics@1"

#: The only fields this module may ever claim. One per reader, and named here
#: so a reader cannot quietly claim a second and become a compiler.
AUTHORS = frozenset({"trigger_semantics", "stated_weights", "day_rule"})


#: Words that invert a condition. The reader has no rule for what a negated
#: trigger means, so it declines rather than reading through them.
_NEGATIONS = frozenset({"not", "n't", "never", "no", "unless"})


def _both_readings_present(parse) -> bool:
    """Whether the sentence contains a transition *and* a state construction.

    Checked on the parse rather than inferred from how many candidates came
    back, because candidate multiplicity turned out not to detect it. "crosses
    below and stays below" binds its level to one governing verb, so exactly
    one family fires and the pair looks unanimous — a sentence carrying both
    readings arriving as a confident single reading, which is the shape this
    reader exists to refuse.
    """
    from .semantics import _DYNAMIC, _STATE

    lemmas = {getattr(t, "lemma", "").lower()
              for sentence in getattr(parse, "sentences", ())
              for t in getattr(sentence, "tokens", ())}
    return bool(lemmas & _DYNAMIC) and bool(lemmas & (_STATE | {"be"}))


def _is_negated(parse) -> bool:
    """Whether any negation attaches to a verb in this sentence.

    Deliberately the whole sentence rather than the trigger clause alone. A
    narrower scope would need the reader to decide which clause the trigger
    belongs to, which is exactly the kind of judgement it must not make — and
    the failure directions are not symmetric. Declining a negated sentence the
    reader could have read costs a question; reading through a negation
    executes the opposite trigger.

    Found by falsification, not by design: `"buy VOO when SPY does not fall
    below its 200-day moving average"` produced `crossing_event`, because the
    dynamic verb and the comparison preposition are both present and nothing
    looked at `not`. Harmless while the derivation was evidence; an
    authoritative wrong answer the moment a reader authored it.
    """
    for sentence in getattr(parse, "sentences", ()):  # noqa: B007
        for token in getattr(sentence, "tokens", ()):
            if getattr(token, "lemma", "").lower() in _NEGATIONS:
                return True
    return False


def trigger_semantics(candidates: Sequence, parse=None,
                      text: str = "") -> Optional[Proposal]:
    """The one claim, or silence.

    Reads the candidates `semantics.propose` already derives — `crossing_event`
    when a dynamic verb governs a comparison, `persistent_condition` when a
    copula does — rather than re-deriving them from the parse. Those rules were
    written against attested sentences and have their own tests; a second
    implementation here would be a second opinion about the same grammar.

    Silence is returned whenever the sentence is not structurally decisive:

        "crosses below"                 -> crossing_event
        "while it is below"             -> persistent_condition
        "crosses below and stays below" -> nothing, because both fire
        "does not fall below"           -> nothing, because it is negated

    The two silences matter more than the two readings. An event and a state
    differ in how often a strategy fires, and guessing between them produced a
    4.6x error in contributed capital the last time they were conflated — so a
    sentence carrying both, or inverting one, is not a sentence this reader
    gets to pick for.
    """
    values = {c.value for c in candidates
              if getattr(c, "field", None) == "trigger_semantics"}
    if len(values) != 1:
        return None
    if parse is not None and (_is_negated(parse)
                              or _both_readings_present(parse)):
        return None
    return Proposal(dimension="trigger_semantics", value=values.pop(),
                    reader_id=TRIGGER_READER_ID)


WEIGHTS_READER_ID = "quantify-weight-binding@1"

#: `60% in VTI`, `40% VTI`, `25% into BND`. A percentage and the holding it is
#: attached to, in that order, with at most a preposition between them.
#:
#: Deliberately narrow. A bare ratio — "a 60/40 portfolio", "rebalance to 70/30"
#: — carries no binding at all, and the sentences in the corpus that state one
#: name no holdings whatever. Pairing those positionally would mean deciding
#: that the first number belongs to the first instrument, which is a coin toss
#: on a dimension where getting it backwards runs 40/60 under the name 60/40:
#: a wrong executable meaning, the class this project spends most of its effort
#: refusing to produce.
_ADJACENT_WEIGHT = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:in|into|to|of|toward|towards)?\s+"
    r"([A-Z][A-Z0-9.\-]{1,9})\b")


def weight_binding(candidates: Sequence, parse=None,
                   text: str = "") -> Optional[Proposal]:
    """Which holding each stated weight belongs to, or silence.

    Read from the sentence rather than from the parse, because the deployment
    that serves users has no deterministic parser installed — a reader that
    needed one would be correct in the suite and absent in production, which is
    the shape of every gap this project has found in its own deployment.

    Returns the binding as `TICKER=weight` pairs so the compiler receives an
    answer rather than a fact it has to re-derive. Silence when fewer than two
    holdings carry a weight: one weight is not a split, and a split nobody can
    attach to a holding is `stated_weights` with no relation — which is exactly
    what the manifest refuses and should keep refusing.
    """
    if not text:
        return None

    pairs = []
    for weight, ticker in _ADJACENT_WEIGHT.findall(text):
        if ticker in dict(pairs):
            # The same holding weighted twice is a sentence nobody can execute
            # without deciding which mention wins.
            return None
        pairs.append((ticker, weight))

    if len(pairs) < 2:
        return None
    return Proposal(dimension="stated_weights",
                    value=",".join(f"{t}={w}" for t, w in pairs),
                    reader_id=WEIGHTS_READER_ID)


DAY_READER_ID = "quantify-day-of-month@1"

#: A day of the month, stated as an ordinal: "the 15th", "on the 3rd".
#:
#: Anchored on `the` or `on the` and requiring the ordinal suffix, because the
#: neighbours are all bare numbers and the cost of confusing them is a plan
#: that runs on a date nobody named:
#:
#:   "$200 into NVDA"            an amount
#:   "the past 5 years"          an evaluation period
#:   "its 200-day moving average" a window
#:   "every month"               a cadence
#:
#: None of those wears an ordinal suffix, and the suffix is what this reads.
_DAY_OF_MONTH = re.compile(
    r"\b(?:on\s+)?the\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

#: Ordinals that name a position in a sequence rather than a date. "the 1st of
#: the month" is a day; "the 1st trading day" is the first-session rule, which
#: this build already executes and which this reader must not overwrite.
_NOT_A_DATE = re.compile(
    r"\b(?:on\s+)?the\s+\d{1,2}(?:st|nd|rd|th)\s+"
    r"(?:trading|business|market|session|of\s+(?:those|these))\b",
    re.IGNORECASE)


def day_of_month(candidates: Sequence, parse=None,
                 text: str = "") -> Optional[Proposal]:
    """The day of the month a contribution lands on, or silence.

    The vocabulary could not state one. Somebody who wrote "on the same day
    each month - the 15th" had it read as `calendar_first_rolled_forward` — the
    *first* of the period — and was then refused for asking for something they
    had not asked for. A reading that drops the day and substitutes a different
    rule is worse than no reading: it puts a plan on the record that the person
    never described.

    Silent unless exactly one day is named. Two ordinals in a sentence is a
    schedule this cannot resolve — "the 1st and the 15th" is twice a month, not
    a day — and choosing one of them would be the coin toss this project
    exists to refuse.
    """
    if not text:
        return None
    if _NOT_A_DATE.search(text):
        return None

    days = {int(found) for found in _DAY_OF_MONTH.findall(text)}
    if len(days) != 1:
        return None
    day = days.pop()
    if not 1 <= day <= 31:
        return None
    return Proposal(dimension="day_rule", value=f"calendar_day:{day}",
                    reader_id=DAY_READER_ID)


#: Every derived reader, so the pipeline does not name them one at a time and
#: a new one cannot be added without appearing in the structural test.
DERIVED_READERS = ((TRIGGER_READER_ID, trigger_semantics),
                   (WEIGHTS_READER_ID, weight_binding),
                   (DAY_READER_ID, day_of_month))
