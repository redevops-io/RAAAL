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
def weights(raw: Any):
    """A stated split, as an unordered set of shares on one scale.

    `60/40`, `VTI=60,BND=40` and the pair `0.6` `0.4` are the same split. The
    first says the shares, the second says which holding takes which, and the
    third is how the deterministic reader emits them — one observation per
    share. Compared on the shares alone, so a reading that carries the binding
    is recognised as the same reading and can be preferred for carrying it.

    Shares above one are read as percentages and divided down, because `60` and
    `0.6` are the same share written two ways and a comparison that called them
    different would report a disagreement between two readers that agree.
    """
    import re
    from decimal import Decimal, InvalidOperation

    shares = []
    for token in re.split(r"[,;/\s]+", str(raw).strip()):
        if not token:
            continue
        try:
            share = Decimal(token.split("=")[-1].rstrip("%"))
        except (InvalidOperation, ValueError):
            return None
        shares.append(share / 100 if share > 1 else share)
    return frozenset(shares) if shares else None


#: Dimensions whose value is assembled from several observations rather than
#: read whole, and how the parts join. A reader emitting one observation per
#: share is not disagreeing with itself; it is describing one value in pieces.
#:
#: This is the seam the two-witness attempt was missing. Without it the
#: deterministic reader's `0.6` and `0.4` reached fusion as two claims about
#: `stated_weights` and were compared individually against the model's `60/40`,
#: which reported a contradiction between readers that agreed.
AGGREGATED = {"SET": ", ", "WEIGHTS": "/"}


def one_claim_per_dimension(observations, *, value_of=None, dimension_of=None):
    """Several observations of one dimension, joined into one claim.

    The step that has to happen before generic fusion, and the reason it lives
    here: whether several observations are one value in pieces or several
    competing answers is a fact about the dimension, and fusion must not have
    to know. It compares one claim per reader per dimension and stays free of
    finance.

    Dimensions not in `AGGREGATED` are untouched. Two observations of a scalar
    genuinely compete, and joining them would invent a value nobody stated.
    """
    value_of = value_of or (lambda o: getattr(o, "value", o))
    dimension_of = dimension_of or (lambda o: getattr(o, "dimension", ""))

    modes = compare_modes()
    parts: Dict[str, list] = {}
    order: list = []
    passthrough = []
    for one in observations:
        name = dimension_of(one)
        joiner = AGGREGATED.get(modes.get(name, "TEXT"))
        if joiner is None:
            passthrough.append(one)
            continue
        if name not in parts:
            parts[name] = []
            order.append(name)
        for token in str(value_of(one)).split(","):
            token = token.strip()
            if token and token not in parts[name]:
                parts[name].append(token)

    joined = []
    for name in order:
        first = next(o for o in observations if dimension_of(o) == name)
        joiner = AGGREGATED[modes.get(name, "TEXT")]
        joined.append(_replaced(first, joiner.join(parts[name])))
    return passthrough + joined


#: Mode name -> what the mode means here. `TEXT` and `SET` are deliberately
#: absent: the runtime implements both, and supplying our own would replace a
#: generic rule with a domain one that happens to agree today.
#: Mode name -> what the mode means here. `TEXT` and `SET` are deliberately
#: absent: the runtime implements both, and supplying our own would replace a
#: generic rule with a domain one that happens to agree today.
NORMALIZERS: Mapping[str, Callable[[Any], Any]] = {"NUMBER": number,
                                                   "WEIGHTS": weights}


def compare_modes() -> Dict[str, str]:
    """Every dimension's comparison mode, read from the schema.

    Read rather than restated. `Dimension.compare_as` has been in the schema
    since the first shadow run and a table here would be a second answer to a
    question the schema already answers.
    """
    # `REQUIREMENTS` first, the schema second, and they disagree for exactly
    # one dimension: `stated_weights` is WEIGHTS in the requirement and SET in
    # the schema. Fusion has always read the requirement, so reading the schema
    # here silently compared `60/40` against `VTI=60,BND=40` as unordered
    # tokens, called them different, and refused a split the compiler had been
    # handed.
    #
    # Two sources for one fact, which is its own defect and is not this
    # adapter's to resolve — but the one fusion uses is the one that must be
    # passed, or the comparison changes meaning at the boundary.
    from .vocabulary import REQUIREMENTS

    modes = {d.name: getattr(d, "compare_as", "TEXT")
             for d in QUANTIFY_SCHEMA.dimensions}
    for name, requirement in REQUIREMENTS.items():
        declared = getattr(requirement, "compare_as", "")
        if declared:
            modes[name] = declared
    return modes


def compare_as(dimension: str) -> str:
    """One dimension's mode, defaulting to exact.

    `TEXT` for an unknown dimension, which reports a difference a rule might
    have reconciled rather than inventing an agreement.
    """
    found = QUANTIFY_SCHEMA.dimension(dimension)
    return getattr(found, "compare_as", "TEXT") if found is not None else "TEXT"


def one_reading_per_set_dimension(readings):
    """One reading per SET dimension. Kept as the name callers already use.

    Delegates to `one_claim_per_dimension`, which answers the same question for
    every aggregated dimension rather than only for sets: are these several
    observations of one value, or several competing answers? Two functions
    answering that is how they would come to disagree — which is the defect
    this whole seam exists to remove, one level up.
    """
    return one_claim_per_dimension(readings)


def _replaced(reading, value):
    """The observation with a new value, whatever concrete type it is.

    Two attribute names, because two kinds of observation flow through here: a
    reader's `Reading` carries `value` and a deterministic candidate carries
    `proposed_value`. Replacing the wrong one silently leaves the original
    value in place, which is an aggregation that reports success and changes
    nothing.
    """
    import dataclasses

    attribute = "proposed_value" if hasattr(reading, "proposed_value") else "value"

    if dataclasses.is_dataclass(reading):
        try:
            return dataclasses.replace(reading, **{attribute: value})
        except Exception:                                      # noqa: BLE001
            pass

    class _Observation:
        pass

    copy = _Observation()
    for attr in ("dimension", "value", "proposed_value", "source_span",
                 "score", "features", "sentence_id", "parser", "model",
                 "scoring_version"):
        if hasattr(reading, attr):
            setattr(copy, attr, getattr(reading, attr))
    setattr(copy, attribute, value)
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


# --- the two-witness profile ------------------------------------------------
#
# The model proposes; syntax argues. This asymmetry is product semantics, not
# an implementation detail, and it is the one thing that must not be lost in
# the move to a runtime whose default stance is that no reader is privileged.
#
#     model + syntax agree        settle, syntax recorded as supporting evidence
#     model + syntax contradict   unresolved: the words argue with the reading
#     model silent, syntax speaks unresolved: syntax alone never carries a field
#     model speaks, syntax silent settle: silence is not an argument
#
# The two middle rows are asymmetric on purpose. A deterministic candidate is
# evidence *about* the model's reading rather than a competing reading, so it
# can contradict a value and cannot supply one. Passing syntax to the runtime
# as a second reader would invert that: it would become a peer proposal source
# and could carry a field alone, which is a different product.

def two_witness_readings(model_reading, syntax_evidence):
    """One `Reading` where syntax argues with the model rather than competing.

    `syntax_evidence` is keyed by dimension — the deterministic candidates for
    each — and is folded in three ways, none of which lets it propose:

      agreeing        appended as `DecisionEvidence` on the model's field, so
                      the record says two witnesses concurred
      contradicting   left out of the payload, so `fuse` sees a field whose
                      value the words argue with
      model-silent    not in the payload at all, so `fuse` receives a
                      dimension with no proposals and returns DISAGREE, which
                      is exactly "syntax alone never carries a field"

    Returned as a single reading rather than two, because two readings is how
    the runtime models two *peers*.
    """
    from runtime_contracts import DecisionEvidence, ReaderKind

    from discovery_runtime import Reading

    reader_id = getattr(model_reading, "reader_id", "model")
    payload, evidence, contradicted = {}, {}, {}

    proposed = {}
    for one in one_reading_per_set_dimension(model_reading.readings):
        proposed[one.dimension] = one

    for name, one in proposed.items():
        payload[name] = one.value
        evidence[name] = [DecisionEvidence(
            reader_id=reader_id, kind=ReaderKind.MODEL, value=one.value,
            source_ref=str(getattr(one, "source_span", "") or ""))]

    modes = compare_modes()
    for name, candidates in (syntax_evidence or {}).items():
        # One claim per reader per dimension, before any comparison. The
        # deterministic reader emits one observation per share — `0.6`, `0.4`
        # for a 60/40 split — and comparing each against the model's `60/40`
        # individually reported a contradiction between two readers that agree.
        # Whether several observations are one value in pieces or several
        # competing answers is a fact about the dimension, not about fusion.
        candidates = one_claim_per_dimension(
            candidates,
            value_of=lambda c: getattr(c, "proposed_value", None),
            dimension_of=lambda c, _name=name: _name)
        for candidate in candidates:
            value = getattr(candidate, "proposed_value", None)
            span = str(getattr(candidate, "source_span", "") or "")
            if name not in proposed:
                # Syntax speaks where the model did not. Recorded so the
                # decision can say what argued, and deliberately not added to
                # the payload: an unproposed dimension reaches fuse with no
                # proposals and comes back DISAGREE.
                evidence.setdefault(name, []).append(DecisionEvidence(
                    reader_id="syntax", kind=ReaderKind.RULE, value=value,
                    source_ref=span))
                continue
            agrees = same_value_for(name, proposed[name].value, value, modes)
            if agrees:
                evidence[name].append(DecisionEvidence(
                    reader_id="syntax", kind=ReaderKind.RULE, value=value,
                    source_ref=span))
            else:
                contradicted[name] = (
                    f"the words say {value!r} where the reading says "
                    f"{proposed[name].value!r}")
    return Reading(payload=payload, evidence=evidence), contradicted


def same_value_for(dimension, one, other, modes=None):
    """The schema's comparison rule, applied by name."""
    from discovery_runtime import same_value

    modes = modes or compare_modes()
    return same_value(one, other, modes.get(dimension, "TEXT"),
                      normalizers=NORMALIZERS)


# --- decisions, produced by the official runtime -----------------------------

class Witnessed:
    """What spoke for a dimension: the value and who said it.

    `witnesses.record` stores `decision.model.reader_id`, so the shim needs a
    holder with that attribute. Deliberately not `fusion.Proposal` — that type
    lives in the module being deleted, and depending on it here would put the
    generic implementation back on the serving path through the back door.
    """

    __slots__ = ("value", "reader_id", "source_span")

    def __init__(self, value, reader_id, source_span=""):
        self.value = value
        self.reader_id = reader_id
        self.source_span = source_span


class RuntimeDecision:
    """An upstream `Decision`, in the shape Quantify's recorder reads.

    `witnesses.record` needs `.model` and `.syntax` to say which readers spoke,
    and the runtime's decision carries proposals rather than named witnesses —
    correctly, since which witness is which is domain knowledge. This adds the
    two attributes back from what the adapter already knows, and forwards the
    rest.

    A shim rather than a conversion so `.outcome` stays the runtime's enum: the
    provenance strings Quantify stores are derived from the outcome's *name*,
    and both enums use the same four names for the same four situations.
    """

    __slots__ = ("dimension", "outcome", "value", "detail", "material",
                 "model", "syntax", "policy_version")

    def __init__(self, decision, *, model=None, syntax=(), material=True):
        self.dimension = decision.dimension
        self.outcome = decision.outcome
        self.value = decision.value
        self.detail = decision.detail
        self.material = material
        self.model = model
        self.syntax = tuple(syntax)
        self.policy_version = "discovery-runtime"

    @property
    def proceeds(self) -> bool:
        return self.outcome.proceeds


def decisions_via_runtime(model_reading, *, syntax_evidence=None,
                          derived=None):
    """Every dimension's decision, made by discovery-runtime.

    One function for both profiles. `syntax_evidence` empty is the
    single-witness profile, and that is not a special case: a profile with no
    second witness is one where nothing argues, which is what an empty mapping
    already means.

    Derived readers are ordinary proposals. Quantify's own fusion says so —
    "a derived reader is a reader, weighed by the ordinary rules" — so they
    join the model's proposals rather than getting a channel of their own.
    """
    from discovery_runtime import Proposal, fuse

    reading, contradicted = two_witness_readings(
        model_reading, syntax_evidence or {})
    modes = compare_modes()
    reader_id = getattr(model_reading, "reader_id", "model")

    proposed = {r.dimension: r
                for r in one_reading_per_set_dimension(model_reading.readings)}

    dimensions = set(reading.payload) | set(reading.evidence)
    dimensions |= set(derived or {})

    out = []
    for name in sorted(dimensions):
        proposals = []
        if name in reading.payload:
            proposals.append(Proposal(value=reading.payload[name],
                                      reader_id=reader_id))
        supplied = (derived or {}).get(name)
        if supplied is not None:
            proposals.append(Proposal(value=supplied.value,
                                      reader_id=getattr(supplied, "reader_id",
                                                        "derived")))
            # The derived reading first when the two agree, because `fuse`
            # settles on the first proposal and the derived one is the richer
            # reading: `60/40` and `VTI=60,BND=40` agree about the split and
            # only the second says which holding takes which share. Keeping the
            # model's value discards the binding and leaves the compiler
            # refusing a split it was handed.
            #
            # Only when they agree. `same_value` establishing that the two are
            # the same reading is what makes preferring the richer one safe: it
            # cannot change what the plan means, only how much of it survives.
            if (name in reading.payload
                    and same_value_for(name, reading.payload[name],
                                       supplied.value, modes)):
                proposals.reverse()
        decision = fuse(name, proposals, mode=modes.get(name, "TEXT"),
                        normalizers=NORMALIZERS,
                        contradicted_by=contradicted.get(name),
                        ambiguous_between=tuple(
                            ambiguity(name, reading.evidence.get(name, ()),
                                      tuple(dimensions))))
        spoke = proposed.get(name)
        out.append(RuntimeDecision(
            decision,
            model=(Witnessed(spoke.value, reader_id,
                             str(getattr(spoke, "source_span", "") or ""))
                   if spoke is not None else
                   (Witnessed(supplied.value,
                              getattr(supplied, "reader_id", "derived"))
                    if supplied is not None else None)),
            syntax=tuple(syntax_evidence.get(name, ())
                         if syntax_evidence else ()),
            material=material(name)))
    return out


def deterministic_witness(text: str, parse, *, language: str = "en"):
    """The deterministic reading: syntax evidence, and what it derives.

    Both come from one pass because both come from the same candidates, and
    computing them separately means normalising and binding the sentence twice
    — which is not only wasted work but two chances to disagree.

    Derived readers see the candidates and the parse here. In the
    single-witness profile they see neither, and that difference is the
    profile: a derived reader that needs structure has none to read when no
    parse was produced.
    """
    from .binding import bind
    from .derived_readers import DERIVED_READERS
    from .pipeline import as_evidence
    from .semantics import propose
    from .syntax import normalize

    values = normalize(text, language)
    candidates = propose(bind(parse, values), values)

    evidence: Dict[str, list] = {}
    for candidate in candidates:
        if getattr(candidate, "is_contract_field", False):
            evidence.setdefault(candidate.field, []).append(
                as_evidence(candidate))

    derived = {}
    for _reader_id, derive in DERIVED_READERS:
        found = derive(candidates, parse, text)
        if found is not None:
            derived[found.dimension] = found
    return evidence, derived


def syntax_evidence_for(text: str, parse, schema=None, *, language: str = "en"):
    """The deterministic witness's candidates, by dimension.

    Lifted out of `pipeline.read` because the pipeline is generic lifecycle and
    goes away, while *this* is Quantify's own deterministic reading: normalise
    the written values, bind them to what the parse says they belong to, and
    propose contract fields from the result. All three steps live in kept
    modules; only the orchestration moved.

    Only contract fields are returned. An intermediate — `amount_kind`,
    `holding_period_days` — has no contract field for the other witness to
    answer with, so offering it to fusion would report a disagreement against a
    silence that could never have been anything else.
    """
    from .binding import bind
    from .pipeline import as_evidence
    from .semantics import propose
    from .syntax import normalize

    candidates = propose(bind(parse, normalize(text, language)),
                         normalize(text, language))
    found: Dict[str, list] = {}
    for candidate in candidates:
        if not getattr(candidate, "is_contract_field", False):
            continue
        found.setdefault(candidate.field, []).append(as_evidence(candidate))
    return found
