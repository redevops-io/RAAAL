"""Today's compiler, speaking the Discovery → Mission contract.

The migration puts Mission before the model, and this module is why that is
possible: the existing regex compiler already produces every field a
`VerifiedIntent` needs, so the contract can be exercised — and the capability
manifest gated against it — with no model in the loop at all.

**What this is not.** It is not a Discovery Runtime. It reads with the same
ordered regex tables that made every new phrasing a code change, and it will be
replaced. What it does is prove the *shape* carries what the engine needs, and
stamp `produced_by: quantify-compiler@N` on everything it emits — so when
Discovery replaces it, the two eras of intent are distinguishable in the record
instead of merged. That stamp is the reason to do this now rather than after
the swap, when it would mean guessing which intents came from which reader.

**Author is the point.** The compiler already separates three things and the
contract keeps them apart:

    recognitions   the user's own words carried it, with the span to prove it
                   -> Author.USER
    inferred       a declared default supplied it and nobody asserted it
                   -> Author.DEFAULT
    unresolved     a question was raised and not answered
                   -> Unresolved(NOT_ASKED), which is not the same as absent

The middle row is the `execution_timing` defect made structural. That value was
inferred, offered back to the user as something to confirm, and thereby became
indistinguishable from a choice they had made. `Author.DEFAULT` is the field
that stops a consumer doing it again.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from runtime_contracts import (
    Author,
    DecisionEvidence,
    IntentField,
    OpenReason,
    ReaderKind,
    Unresolved,
    VerifiedIntent,
)
from runtime_contracts.canonical import decimal_string

#: Stamped on every intent this reader produces.
#:
#: Bumped when the *reading* changes in a way that could turn one sentence into
#: a different intent — a new pattern, a changed precedence — and not when
#: unrelated code moves. A version that changes for other reasons makes the
#: field useless for the one job it has: telling you which intents came from a
#: reader with a known bug.
READER_VERSION = "quantify-compiler@1"

#: `contribution_day_rule` is the compiler's name for it; `day_rule` is the
#: engine's and the manifest's. Renamed here, at the boundary, so neither side
#: has to learn the other's spelling.
_RENAME = {"contribution_day_rule": "day_rule"}


def _named(field: str) -> str:
    return _RENAME.get(field, field)


def from_compiled(result, parsed, *, objective: str = "evaluate_investment_strategy",
                  utterance_ref: str = "",
                  created_at: Optional[str] = None) -> VerifiedIntent:
    """A `VerifiedIntent` from a compiled plan and the parse behind it.

    Both are needed and neither is enough: `parsed.recognitions` says what the
    user's words carried, `result.inferred` says what the compiler supplied,
    and only together do they say who authored each field.
    """
    fields: Dict[str, IntentField] = {}

    # The user's words first, so a stated value is never overwritten by an
    # inference for the same field. USER dominates, and this is where that
    # starts being true.
    for recognition in getattr(parsed, "recognitions", ()) or ():
        name = _named(recognition.field)
        fields[name] = IntentField(
            value=recognition.value,
            author=Author.USER,
            produced_by=READER_VERSION,
            source_span=str(recognition.span or ""),
            evidence=(DecisionEvidence(
                reader_id=READER_VERSION, kind=ReaderKind.RULE,
                value=recognition.value,
                source_ref=str(recognition.span or "")),),
        )

    for inference in getattr(result, "inferred", ()) or ():
        name = _named(inference.field)
        if name in fields:
            # Stated beats inferred. Silently letting a default win here is the
            # authority inversion this contract exists to make impossible.
            continue
        fields[name] = IntentField(
            value=inference.value,
            author=Author.DEFAULT,
            produced_by=READER_VERSION,
            evidence=(DecisionEvidence(
                reader_id=READER_VERSION, kind=ReaderKind.PRIOR,
                value=inference.value,
                source_ref=str(getattr(inference, "why", "") or "")),),
        )

    # Declarations the user made that never become a `Recognition`.
    #
    # Found by building this: the compiler reads a stated 60/40 through
    # `stated_weights(text)` and coverage reads unsupported weighting straight
    # from the prose, so neither reaches `recognitions` — and an intent without
    # them cannot be refused by the manifest, because Mission never learns they
    # were asked for. Coverage still blocks the figure by its own path, so the
    # product was safe and the *contract* was incomplete, which is precisely
    # the sort of gap a boundary artifact exists to expose.
    #
    # A real Discovery Runtime will emit these as ordinary fields. Until then
    # they are bridged here rather than left out, because "the intent did not
    # mention it" and "the user did not ask for it" must not be the same thing.
    fields.update(_from_prose(getattr(parsed, "text", "") or "", fields))

    # A question raised and not answered is *open*, not absent. Collapsing the
    # two is how a default gets applied to a dimension the user was still being
    # asked about.
    unresolved = tuple(
        Unresolved(dimension=_named(getattr(one, "field", str(one))),
                   reason=OpenReason.NOT_ASKED,
                   detail="raised by the compiler and not yet settled")
        for one in (getattr(result, "unresolved", ()) or ())
        if _named(getattr(one, "field", str(one))) not in fields)

    return VerifiedIntent(
        objective=objective,
        fields=fields,
        unresolved=unresolved,
        produced_by=READER_VERSION,
        utterance_ref=utterance_ref,
        created_at=created_at,
    )


def _from_prose(text: str, already: Dict[str, IntentField]) -> Dict[str, IntentField]:
    """Dimensions the user stated in prose that never became a `Recognition`.

    Each is `Author.USER` — they are the user's words — and each carries the
    phrase that carried it, so a refusal can quote them back.
    """
    from .compiler import stated_weights
    from .coverage import _PERIODIC_REBALANCING, _SELL_LEG, _UNSUPPORTED_WEIGHTING

    found: Dict[str, IntentField] = {}
    if not text:
        return found

    def add(name: str, value: Any, span: str) -> None:
        if name in already or name in found:
            return
        found[name] = IntentField(
            value=value, author=Author.USER, produced_by=READER_VERSION,
            source_span=span,
            evidence=(DecisionEvidence(reader_id=READER_VERSION,
                                       kind=ReaderKind.RULE, value=value,
                                       source_ref=span),))

    # Assets, the watched series and the average length. All three are things
    # the compiler reads and none is a `Recognition`, so an intent built from
    # recognitions alone names nothing to hold — and a plan compiled from it
    # holds nothing.
    #
    # `src/discovery/readers_quantify.py` bridges the same three for the shadow
    # comparison. The duplication is deliberate and temporary: this module is
    # Phase 2 scaffolding that dies when the legacy compiler is deleted, and
    # collapsing the two now would mean maintaining a merge of a thing that is
    # going away.
    from .compiler import moving_average_window
    from .time_window import detect as detect_window

    parsed = _parse_once(text)
    if parsed is not None:
        if parsed.assets:
            add("assets", ", ".join(parsed.assets), "")
        if parsed.observed and set(parsed.observed) != set(parsed.assets):
            add("observed_assets", ", ".join(parsed.observed), "")
    window = moving_average_window(text)
    if window:
        add("moving_average_window", str(window), "")

    weights = stated_weights(text)
    if weights:
        # As decimal strings, because the canonical form refuses floats:
        # Python, Go and JavaScript do not print the same digits for the same
        # double, so a float here hashes differently per runtime. Caught by the
        # contract on the first run of this test file, which is the contract
        # doing its job on its first customer.
        add("stated_weights", tuple(decimal_string(str(w)) for w in weights), "")

    for name, pattern in (("allocation_method", _UNSUPPORTED_WEIGHTING),
                          ("periodic_rebalancing", _PERIODIC_REBALANCING),
                          ("sell_action", _SELL_LEG)):
        match = pattern.search(text)
        if match:
            phrase = _clause(match.group(0))
            add(name, phrase, phrase)
    return found


def _parse_once(text: str):
    from .compiler import parse

    try:
        return parse(text)
    except Exception:                                             # noqa: BLE001
        return None


def _clause(matched: str) -> str:
    """The user's phrase, cut at the clause boundary.

    These patterns are written to *detect*, not to delimit, so a match can run
    on past the thing it found — "rebalance quarter end, over the past 5 years"
    is one declaration and one unrelated clause. Quoting the whole span back at
    a user as what they asked for misdescribes their sentence in a message
    whose entire job is to describe it accurately.
    """
    return matched.split(",")[0].strip()


def executable_check(intent: VerifiedIntent) -> tuple:
    """Every refusal this build owes the reader for this intent.

    The Mission half of the split in one call: the intent is read and never
    edited, and each dimension the manifest cannot execute produces a named
    refusal rather than a substitution.

    Returns *all* of them deliberately. A user who described three unsupported
    things should learn that once rather than across three attempts.
    """
    from .capability import Refusal, refusals_for

    declared = {name: f.value for name, f in intent.fields.items()}
    refusals = list(refusals_for(declared))

    # An unresolved disagreement is a refusal too, and a different one: the
    # engine could run it, and nobody has said what "it" is.
    for open_dimension in intent.blocking:
        refusals.append(Refusal(
            kind="UNRESOLVED_INPUT",
            dimension=open_dimension.dimension,
            detail=("readers disagreed and it was not settled, so running it "
                    "would mean choosing a reading nobody chose")))
    return tuple(refusals)


def derivation(intent: VerifiedIntent, *, compiled_by: str) -> Dict[str, Any]:
    """The provenance a compiled plan carries back to its intent.

    By hash, not by reference: an intent that changed is a different intent,
    and a plan pointing at a mutable id would silently re-describe itself.
    """
    from .capability import MANIFEST_SCHEMA

    return {"compiled_from": intent.intent_hash,
            "compiled_by": compiled_by,
            "manifest_hash": MANIFEST_SCHEMA}
