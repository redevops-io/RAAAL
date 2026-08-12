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

from typing import Optional, Sequence

from .fusion import Proposal

#: Versioned like every other reader. A derived reading whose rules changed
#: under a fixed id would make two runs look comparable when they are not.
TRIGGER_READER_ID = "quantify-trigger-semantics@1"

#: The only field this module may ever claim.
AUTHORS = frozenset({"trigger_semantics"})


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


def trigger_semantics(candidates: Sequence, parse=None) -> Optional[Proposal]:
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


#: Every derived reader, so the pipeline does not name them one at a time and
#: a new one cannot be added without appearing in the structural test.
DERIVED_READERS = ((TRIGGER_READER_ID, trigger_semantics),)
