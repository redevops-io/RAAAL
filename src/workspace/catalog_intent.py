"""A picked strategy, straight to a sealed intent, with no reader involved.

    CatalogSelection + CatalogDefaults + UserEdits -> VerifiedIntent

The free-text path and this one meet at `VerifiedIntent` and nowhere earlier.
That is the whole design: everything downstream — Mission, the specification,
the evaluator — sees one artifact and cannot tell which door it came through,
while the two doors keep their own account of *who said what*.

**Three authorities, and they do not collapse.**

    MODEL    a reading of somebody's words          (free text only)
    READER   the catalogue states it, as itself     (a picked entry)
    DEFAULT  nobody said it; the family supplied it (assumed, not stated)
    USER     they typed it, or accepted it          (edits, either path)

A catalogue value is `READER`, not `MODEL`: no model was consulted. It is not
`USER` either — the person chose the entry, not the value, and `Author.USER`
dominates every other author and is never overwritten by a re-read. Recording
our template as their word would make it permanent and invisible, which is the
same inversion the assumed-value work was built to prevent.

**No model, and that is checkable.** `intent_for` takes no reader and imports
none. `tests/test_catalog_structured_path.py` runs it with the hosted reader
patched to raise, which is the only version of "does not call a model" that a
test can hold.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from runtime_contracts import Author, IntentField, VerifiedIntent

from ..discovery.canonical import canonicalise

#: Where a sealed intent says it came from. Carried into `produced_by`, so a
#: stored plan can say it was never read by a model — which is a different
#: provenance claim from "a model read it and agreed", and the two must not
#: look alike in the record.
CATALOG_VERSION = "quantify-catalog@1"

_AUTHORS = {"MODEL": Author.MODEL, "READER": Author.READER,
            "USER": Author.USER, "DEFAULT": Author.DEFAULT}


def intent_for(entry_key: str, *, edits: Optional[Mapping[str, Any]] = None,
               objective: str = "evaluate_investment_strategy"
               ) -> Tuple[Optional[VerifiedIntent], Sequence[Tuple[str, str]]]:
    """The sealed intent a selection means, and anything unreadable in it.

    Returns `(None, ())` when the entry has no structured evidence, so a caller
    can fall back to reading the sentence and *know* that it did. A silent
    fallback would make this feature look complete while half the catalogue
    still went through a model.

    Edits win over the entry, because they are the part the person actually
    typed. They are canonicalised on the way in exactly like anything else —
    the door is different, the form downstream is not.
    """
    from .catalog_evidence import STATES, states, unresolved

    # Membership, not truthiness. `leverage` states nothing its reading could
    # settle, and testing the dict for emptiness reported it as an entry with
    # no structured evidence — so the one strategy the table describes as
    # stating nothing was the one that still went through a model.
    if entry_key not in STATES:
        return None, ()

    stated = dict(states(entry_key))
    supplied = {name: value for name, value in (edits or {}).items()
                if value not in (None, "")}

    # Canonicalised apart, so an unreadable edit loses only itself.
    #
    # Merging first and canonicalising once looked tidier and destroyed the
    # entry's own value: typing "a portion" into the amount overwrote `500`,
    # then failed to canonicalise, and the field vanished from the intent
    # entirely. The person's bad edit took the catalogue's good value with it.
    from_entry = canonicalise(stated)
    from_user = canonicalise(supplied)

    fields = {}
    for name, (value, _author) in from_entry.fields.items():
        fields[name] = IntentField(value=value, author=Author.READER)
    for name, (value, _author) in from_user.fields.items():
        fields[name] = IntentField(value=value, author=Author.USER)

    canonical = from_user            # only an edit can be unreadable here:
    supplied = {name: value for name, value in supplied.items()
                if name in from_user.fields}

    # What the entry leaves open, minus anything the person has now supplied.
    #
    # Without this the structured path sealed every selection and a strategy
    # naming no holding went straight to "the intent names nothing to hold",
    # where picking the same strategy and letting a model read the sentence
    # asked what to hold. Same entry, same words, two different products —
    # which is exactly the drift the equivalence test exists to catch, and it
    # caught it.
    still_open = tuple(one for one in unresolved(entry_key)
                       if one.dimension not in supplied)

    draft = VerifiedIntent(
        objective=objective,
        produced_by=f"{CATALOG_VERSION}+{entry_key}",
        utterance_ref=f"catalog:{entry_key}",
        fields=fields, unresolved=still_open)
    try:
        return draft.seal(), canonical.refusals
    except Exception:                                          # NotSealable
        return draft, canonical.refusals


def reading_for(entry_key: str, text: str, *,
                edits: Optional[Mapping[str, Any]] = None):
    """A `PilotReading` for a picked strategy, or `None` to read the sentence.

    The page and everything under it take a reading, so the structured path
    produces one rather than a second shape nothing renders. `settled` is empty
    on purpose: it records what *fusion* concluded from words, and no words
    were read — the values are in the intent, where a catalogue value is
    `Author.READER` and can be told from a model's proposal.
    """
    from .pilot import PilotReading
    from ..mission.capability import Refusal
    from ..mission.from_intent import NotExecutable, compile_intent

    intent, unreadable = intent_for(entry_key, edits=edits)
    if intent is None:
        return None

    compiled, refusals = None, ()
    if intent.is_verified:
        try:
            compiled = compile_intent(intent)
            refusals = compiled.refusals
        except NotExecutable as refused:
            refusals = refused.refusals

    refusals = tuple(refusals) + tuple(
        Refusal(kind="UNRESOLVED_INPUT", dimension=name, detail=why)
        for name, why in unreadable
        if not any(r.dimension == name for r in refusals))

    return PilotReading(
        text=text, intent=intent, compiled=compiled, settled=(),
        open_fields=tuple(sorted(one.dimension for one in intent.unresolved)),
        absent_fields=(), refusals=refusals,
        reader_id=f"{CATALOG_VERSION}+{entry_key}")


def reads_a_model(entry_key: str) -> bool:
    """Whether choosing this entry still requires a language model.

    Named as a question the product can answer about itself rather than as an
    implementation detail, because it is the measurement Step 2 is judged on
    and it should be countable rather than argued.
    """
    from .catalog_evidence import STATES

    # Membership, for the same reason `intent_for` uses it: an entry may
    # legitimately state nothing this build can settle — `leverage` is one —
    # and "described as stating nothing" is not "undescribed".
    return entry_key not in STATES
