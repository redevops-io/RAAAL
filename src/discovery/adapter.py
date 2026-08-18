"""Quantify's schema, handed to the generic Discovery runtime.

The runtime does not know what a dimension means and must not learn. It knows
that a dimension has a *comparison mode* and that somebody can tell it what a
mode means; this module is the somebody. Everything here is a statement of
finance vocabulary or a lookup into `QUANTIFY_SCHEMA` — there is no comparison
logic, no fusion, no sealing, and a check in `tests/test_discovery_adapter.py`
asserts that, because the whole point of the extraction is that those live in
one place now.

**Three modes, and only one of them is ours.** Across nineteen dimensions the
schema uses `TEXT` (14), `SET` (3) and `NUMBER` (2). `TEXT` is exact and `SET`
is unordered tokens — neither is a domain fact and both are implemented in the
runtime. `NUMBER` is entirely a domain fact: whether `£2.5k` and `2500` are the
same number is a question about money, and answering it needs the normaliser
that already knows `12-month` is a window and `£1k` is a thousand.

**One normaliser, not a second opinion.** `syntax.normalize` is the same
function the deterministic reader uses. A separate numeric parser here would be
a second place deciding what a written number means, which is how the
deterministic path and the model came to disagree about `$500` in the first
place.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Mapping, Optional

from .schema import QUANTIFY_SCHEMA

#: Kinds `syntax.normalize` emits that carry a comparable magnitude. Named
#: rather than "anything with a `canonical`", so a new kind is a decision
#: somebody makes in a diff rather than something that silently starts
#: participating in equality.
NUMERIC_KINDS = ("money", "duration", "percentage", "moving_average_window")


def number(raw: Any) -> Optional[Decimal]:
    """What a written number is worth, by the reader's own rule.

    `None` when it cannot be read, which the runtime treats as "not equal to
    anything" — including to another unreadable value. That is the safe
    direction: two things nobody could parse are not thereby the same thing.
    """
    from .syntax import normalize

    text = str(raw).strip()
    for value in normalize(text):
        if value.kind in NUMERIC_KINDS:
            try:
                return Decimal(str(value.canonical))
            except (InvalidOperation, ValueError):
                return None

    # A bare figure the normaliser does not claim: `500`, `1,000`. Not a
    # fallback that guesses — anything with a suffix or a symbol has already
    # been handled above, and this only strips separators.
    try:
        return Decimal(text.replace(",", "").lstrip("$£€"))
    except (InvalidOperation, ValueError):
        return None


#: Mode name -> what the mode means here. `TEXT` and `SET` are deliberately
#: absent: the runtime implements both, and supplying our own would replace a
#: generic rule with a domain one that happens to agree today.
NORMALIZERS: Mapping[str, Callable[[Any], Any]] = {"NUMBER": number}


def compare_modes() -> Dict[str, str]:
    """Every dimension's comparison mode, read from the schema.

    Read rather than restated. `Dimension.compare_as` has been in the schema
    since the first shadow run and a table here would be a second answer to a
    question the schema already answers.
    """
    return {d.name: getattr(d, "compare_as", "TEXT")
            for d in QUANTIFY_SCHEMA.dimensions}


def compare_as(dimension: str) -> str:
    """One dimension's mode, defaulting to exact.

    `TEXT` for an unknown dimension, which reports a difference a rule might
    have reconciled rather than inventing an agreement.
    """
    found = QUANTIFY_SCHEMA.dimension(dimension)
    return getattr(found, "compare_as", "TEXT") if found is not None else "TEXT"


def one_reading_per_set_dimension(readings):
    """One reading per SET dimension, carrying the whole set.

    A reader must emit one semantic value per SET dimension. Given
    "take from bonds in a down year and from stocks otherwise" the recorded
    reader emits *two* `assets` readings — 'bonds' and 'stocks' — and both
    lanes then got it wrong in different ways: the internal path built
    `{p.dimension: p for p in proposals}` and silently kept the last, so a
    plan for a sentence naming both mentioned only stocks; the runtime read
    two readings from one reader as a disagreement and asked which the person
    meant, of a reader disagreeing with itself.

    They are members, not witnesses. This unions them into the one reading the
    reader should have emitted.

    **Only the membership.** The conditional meaning in that sentence — take
    from bonds *in a down year* — is `sell_action`'s and stays there. Nothing
    here infers a rule from the multiplicity: two members mean two members,
    and if the condition cannot be represented it is that dimension that must
    clarify or refuse, not the asset set.

    Non-SET dimensions are untouched. Two values for a scalar dimension are
    genuinely competing and belong in fusion's hands.
    """
    modes = compare_modes()
    members: Dict[str, list] = {}
    order: list = []
    out = []
    for one in readings:
        name = getattr(one, "dimension", "")
        if modes.get(name) != "SET":
            out.append(one)
            continue
        if name not in members:
            members[name] = []
            order.append(name)
        for token in str(getattr(one, "value", "")).split(","):
            token = token.strip()
            if token and token not in members[name]:
                members[name].append(token)
    for name in order:
        first = next(r for r in readings if getattr(r, "dimension", "") == name)
        out.append(_replaced(first, ", ".join(members[name])))
    return out


def _replaced(reading, value):
    """The reading with a new value, whatever concrete type it is."""
    import dataclasses

    if dataclasses.is_dataclass(reading):
        try:
            return dataclasses.replace(reading, value=value)
        except Exception:                                      # noqa: BLE001
            pass

    class _Reading:
        pass

    copy = _Reading()
    for attr in ("dimension", "value", "source_span"):
        setattr(copy, attr, getattr(reading, attr, ""))
    copy.value = value
    return copy


def ambiguity(dimension, evidence, proposed):
    """Competing readings the *words* carry, or nothing.

    Quantify's `AMBIGUOUS_TERMS` stays here — which terms people demonstrably
    use for two things, and between which dimensions, is finance vocabulary
    with sources attached. The runtime is handed the observation and provides
    the outcome; it never learns the words.

    Two conditions, mirroring `fusion._ambiguity`, and the second is what stops
    this firing on its own ontology. The term must appear in the *person's
    words* — the evidence's source span, never the dimension name, or
    `periodic_rebalancing` matches "rebalance" on every reading it ever
    produces. And a competing dimension must also have been proposed: an
    ambiguity nobody could have resolved differently in this sentence is not
    an ambiguity. "rebalanced annually" carries the word and no ambiguity;
    "rebalance back to 60/40" carries both readings.
    """
    from .vocabulary import AMBIGUOUS_TERMS

    words = " ".join(str(getattr(e, "source_ref", "") or "")
                     for e in evidence).lower()
    seen = set(proposed)
    for term, record in AMBIGUOUS_TERMS.items():
        if term not in words:
            continue
        between = set(record.get("between", ()))
        if between and dimension not in between:
            continue
        if between and not (between - {dimension}) & seen:
            continue
        return tuple(sorted(between)) or (term,)
    return ()


def material(dimension: str) -> bool:
    """Whether leaving this dimension open changes the result.

    Read from `fusion.REQUIREMENTS`, which already carries it — a table here
    would be a second answer to a question Quantify already answers, and the
    two would drift.
    """
    from .vocabulary import REQUIREMENTS, Requirement

    return bool(REQUIREMENTS.get(dimension, Requirement()).material)


def fusion_policy():
    """`merge_readings`, already carrying the schema's rules.

    Returned as a closure rather than asking every caller to remember two
    keyword arguments: a caller that forgot `normalizers` would get exact
    comparison on amounts and a clarification question on every `$500` a reader
    normalised, which is the defect the extraction fixed and the easiest one to
    reintroduce by omission.
    """
    from discovery_runtime import merge_readings

    modes = compare_modes()

    def fuse_readings(readings):
        return merge_readings(readings, compare_as=modes,
                              normalizers=NORMALIZERS, ambiguity=ambiguity,
                              material=material)

    return fuse_readings


def canonicalizer():
    """Quantify's canonicalisation, and a place to keep what it refused.

    `draft_intent` takes a canonicalizer that returns a mapping, so there is no
    channel for "this value could not be canonicalised". The refusals are
    captured in the closure and folded into `unresolved` by `intent_from`.

    They must be, and this is the defect that made the gate red: a stated value
    that cannot be read blocks the seal rather than being dropped. Dropping it
    leaves the dimension *absent*, and absent means the engine may apply its
    default — so an unreadable `4%` would quietly become a plan that runs on a
    number nobody stated.
    """
    from .canonical import canonicalise

    refused: list = []

    def canonicalize(payload):
        refused.clear()
        settled = dict(payload or {})
        result = canonicalise(settled)
        refused.extend(result.refusals)
        return {name: value for name, (value, _author) in result.fields.items()}

    return canonicalize, refused


def runtime(readers, *, objective: str = "evaluate_investment_strategy",
            canonicalize=None):
    """A `DiscoveryRuntime` configured for Quantify.

    The schema travels as `schema` so a caller can introspect its own
    dimensions; the runtime never looks inside it.
    """
    from discovery_runtime import DiscoveryRuntime

    return DiscoveryRuntime(
        readers=list(readers),
        schema=QUANTIFY_SCHEMA,
        objective=objective,
        canonicalize=canonicalize or canonicalizer()[0],
        fusion_policy=fusion_policy(),
    )


# --- who established the value --------------------------------------------
#
# `draft_intent` stamps every field it drafts `Author.READER`, because a
# generic runtime knows a reader produced the value and cannot know what kind
# of reader. Quantify can: the evidence names its `ReaderKind`, and the mapping
# from a kind to an author is a statement about what those witnesses mean here.
#
# It matters beyond tidiness. `author` is inside `canonical_form` and therefore
# inside `intent_hash`, while producer, span and evidence are not — so two
# implementations that classify the same witness differently produce different
# identities for the same request, and an equivalence run would report a
# mismatch that is really this gap.
#
# The internal path already says `MODEL` for a hosted-model reading, so that is
# the classification both paths must agree on.
WITNESS_AUTHORS = {
    "MODEL": "MODEL",        # the hosted reader read it
    "RULE": "READER",        # a deterministic parser read it
    "RETRIEVAL": "READER",   # still a reader, with a different source
    "PRIOR": "DEFAULT",      # an assumption: catalogue or system
    "POLICY": "POLICY",      # imposed by policy, not read at all
    "HUMAN": "USER",         # a structured action by the person
}


def author_for(evidence) -> Any:
    """The author a piece of evidence implies, by its witness kind.

    Defaults to `READER` for an unrecognised kind: something read it and we
    cannot say what. Never `USER` — that is reserved for a structured action,
    and guessing it would hand an unknown witness the one authority a re-read
    can never correct.
    """
    from runtime_contracts import Author

    kind = getattr(getattr(evidence, "kind", None), "value", "")
    return getattr(Author, WITNESS_AUTHORS.get(kind, "READER"))


def classify_authors(intent):
    """Re-author a drafted intent from the witnesses its evidence names.

    Applied after `draft_intent` rather than inside it: the runtime is right
    not to guess, and this is the domain saying what its witnesses are. A field
    with no evidence keeps whatever it arrived with — inventing an author for a
    value nothing witnessed would be worse than leaving the generic one.
    """
    from dataclasses import replace

    fields = {}
    for name, field in intent.fields.items():
        if not field.evidence:
            fields[name] = field
            continue
        fields[name] = replace(field, author=author_for(field.evidence[0]))
    return replace(intent, fields=fields)


# --- Quantify's readers, in the runtime's protocol -------------------------

def as_intent_relation(relation):
    """Quantify's `RelationReading` as the contract's `IntentRelation`.

    The vocabulary boundary. The two shapes correspond almost exactly — kind,
    members, attributes, source_span — and the translation lives here because
    what an `account_transition` *is* stays Quantify's, while the contract only
    needs to know a relation was established, between whom, and by whom.

    `author=READER` rather than the field-level classification: a relation is
    established by whichever reader found it, and no relation reaches this from
    a structured user action today. If one ever does, it arrives with its own
    evidence and this is where that would be read.
    """
    from runtime_contracts import Author, IntentRelation, RelationMember

    members = tuple(
        RelationMember(role=str(role), subject=str(subject),
                       qualifiers=dict(qualifiers or {}))
        for role, subject, qualifiers in getattr(relation, "members", ()) or ())
    return IntentRelation(
        kind=str(getattr(relation, "kind", "")),
        members=members,
        attributes=dict(getattr(relation, "attributes", {}) or {}),
        author=Author.READER,
        produced_by="quantify-binding",
        source_span=str(getattr(relation, "source_span", "") or ""),
    )


def relation_fields(relations) -> Dict[str, str]:
    """Relation kinds as flat markers, so Mission's compiler can refuse them.

    Mirrors `workspace.pilot._relation_fields`. Kept here rather than upstream
    because the need is Mission's flat-field compiler, not the contract — the
    contract carries relations structurally and always has.
    """
    summary: Dict[str, str] = {}
    for relation in relations or ():
        kind = getattr(relation, "kind", "")
        if not kind:
            continue
        # `role=subject` pairs, matching `workspace.pilot._relation_fields`
        # exactly. The marker names what the person described — "from=traditional
        # IRA, to=Roth" — so a refusal can quote it back; the bare kind would
        # tell somebody a relation existed and nothing about which one.
        members = ", ".join(
            f"{role}={subject}" for role, subject, *_ in
            (m if isinstance(m, (tuple, list)) else (m, "", "")
             for m in getattr(relation, "members", ())))
        summary[str(kind)] = members or str(kind)
    return summary


class ReaderAdapter:
    """A Quantify reader, presented as a `discovery_runtime.Reader`.

    The two protocols differ in one way that matters: a Quantify reader returns
    a `ReadingSet` — a flat list of per-dimension readings plus an `ok` flag —
    and the runtime wants a `Reading` whose evidence is *filed under* the
    dimension it supports. That is the same restructuring the contract makes,
    so it happens once, here.

    **A failed read produces no evidence, not empty evidence.** `ReadingSet.ok`
    is false when the reader could not be reached, and a transport failure is
    not a reading: scoring one as silence would let a timeout look like a
    reader that had nothing to say.
    """

    def __init__(self, reader, kind=None, schema=None):
        from runtime_contracts import ReaderKind

        self._reader = reader
        self.reader_id = getattr(reader, "id", None) or getattr(
            reader, "reader_id", "quantify-reader")
        self.kind = kind or ReaderKind.MODEL
        self._schema = schema or QUANTIFY_SCHEMA

    def read(self, text: str):
        from runtime_contracts import DecisionEvidence

        from discovery_runtime import Reading

        reading_set = self._reader.read(text, self._schema)
        if not getattr(reading_set, "ok", True):
            return Reading(payload={})

        payload, evidence = {}, {}
        for one in one_reading_per_set_dimension(reading_set.readings):
            payload[one.dimension] = one.value
            evidence.setdefault(one.dimension, []).append(
                DecisionEvidence(
                    reader_id=getattr(reading_set, "reader_id", self.reader_id),
                    kind=self.kind, value=one.value,
                    source_ref=str(getattr(one, "source_span", "") or "")))

        # Relations, and the flat markers Mission needs to see them.
        #
        # `compile_intent` builds what it asks the manifest about from a flat
        # name -> value map of dimensions, so a relation it cannot see is a
        # refusal that exists in the manifest and never fires. The relation
        # itself stays structured; this adds a marker under the relation's own
        # name so a refusal can name what the person described.
        #
        # The flattening is Quantify's, not the runtime's: it exists because
        # Mission reads flat fields, which is a fact about Mission.
        found = list(getattr(reading_set, "relations", ()) or ())
        markers = relation_fields(found)
        payload.update(markers)
        # Evidence for the markers, naming the reader that established the
        # relation. Without it `classify_authors` has nothing to classify and
        # the marker keeps `draft_intent`'s generic READER, while the internal
        # path says MODEL — a disagreement about who established the value, on
        # a field that is inside canonical_form.
        for name, value in markers.items():
            evidence.setdefault(name, []).append(
                DecisionEvidence(
                    reader_id=getattr(reading_set, "reader_id", self.reader_id),
                    kind=self.kind, value=value,
                    source_ref=str(getattr(
                        next((r for r in found
                              if str(getattr(r, "kind", "")) == name), None),
                        "source_span", "") or "")))
        return Reading(payload=payload, evidence=evidence,
                       relations=[as_intent_relation(r) for r in found])


def intent_from(readers, text: str, *,
                objective: str = "evaluate_investment_strategy"):
    """Draft an intent through the runtime, with the domain's classification.

    The two steps are separate on purpose: `draft_intent` produces a generic
    reading and `classify_authors` says what the witnesses were. Combining them
    upstream would make the runtime guess; leaving them apart here would let a
    caller forget the second and produce an intent whose identity disagrees
    with the internal path for no reason a person could see.
    """
    from runtime_contracts import OpenReason, Unresolved

    canonicalize, refused = canonicalizer()
    intent = classify_authors(
        runtime(readers, objective=objective,
                canonicalize=canonicalize).draft(text))

    if not refused:
        return intent

    # Folded in exactly as the internal path does: a value Discovery cannot
    # canonicalise is result-changing and blocks the seal.
    from dataclasses import replace

    return replace(intent, unresolved=intent.unresolved + tuple(
        Unresolved(dimension=name,
                   reason=OpenReason.UNRESOLVED_DISAGREEMENT,
                   detail=why, result_changing=True)
        for name, why in refused))
