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
                              normalizers=NORMALIZERS)

    return fuse_readings


def runtime(readers, *, objective: str = "evaluate_investment_strategy"):
    """A `DiscoveryRuntime` configured for Quantify.

    The schema travels as `schema` so a caller can introspect its own
    dimensions; the runtime never looks inside it.
    """
    from discovery_runtime import DiscoveryRuntime

    return DiscoveryRuntime(
        readers=list(readers),
        schema=QUANTIFY_SCHEMA,
        objective=objective,
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
