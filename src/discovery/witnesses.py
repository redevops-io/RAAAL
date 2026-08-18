"""Which witnesses a deployment has, and why absence is not agreement.

The pilot runs **model-only**: the hosted reader interprets, and the
deterministic path is not installed. That is a deployment profile and not a
weakened semantic rule, and the difference has to survive into the persisted
artifact or the pilot's evidence is worthless.

    syntax witness unavailable   ≠   syntax witness agrees

`fuse` cannot tell those apart on its own — it receives an empty sequence of
syntax evidence either way. Silence from an installed reader is a fact about
the sentence; silence from a reader nobody installed is a fact about the
container. So the profile is declared here, carried on the decision, and
recorded on the plan.

**What this buys later.** Three populations become separable in the pilot data:

    MODEL_ONLY_ACCEPTED    one witness, and only one was available
    AGREE                  two witnesses, independently
    DISAGREE               two witnesses, and they differed

Without the distinction the first would be indistinguishable from the second,
and a pilot that reported "42 agreements" while running one reader would be
claiming corroboration it never had. That is the classification-without-evidence
defect (§8.3a) at the point where it would be most expensive — in the data used
to decide whether the runtime was worth building.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from .claims import Decision
from discovery_runtime.fusion import Fusion


class Witness(str, Enum):
    MODEL = "model"
    SYNTAX = "syntax"


@dataclass(frozen=True)
class WitnessProfile:
    """The readers this deployment actually has.

    Declared rather than inferred from whether a reader happens to import —
    the same rule `ModelTarget.mode` already applies to the parser, and for the
    same reason: a deployment states what it is, and an entire pilot was once
    measured model-assisted because a variable was set in a shell.
    """

    available: frozenset
    reason: str = ""
    """Why a witness is absent, in one line, for the artifact to carry. "not
    installed in this image" is a different fact from "the model key is
    missing", and a reader of the plan a year from now can act on neither if
    the artifact says only that a witness was silent."""

    def has(self, witness: Witness) -> bool:
        return witness in self.available

    @property
    def is_single_witness(self) -> bool:
        return len(self.available) < 2

    def to_json(self) -> dict:
        return {"available": sorted(w.value for w in self.available),
                "single_witness": self.is_single_witness,
                "reason": self.reason}


#: What the pilot runs. Stanza is deliberately absent from `requirements-core`
#: — the image does not carry a parser model — so the deterministic witness is
#: not merely quiet, it is not there.
MODEL_ONLY = WitnessProfile(
    available=frozenset({Witness.MODEL}),
    reason="the deterministic syntax reader is not installed in this image; "
           "Stanza is in requirements.txt and not requirements-core.txt")

BOTH = WitnessProfile(available=frozenset({Witness.MODEL, Witness.SYNTAX}))


def witnesses_of(decision: Decision) -> tuple:
    """Which readers actually spoke about this decision."""
    speaking = []
    if decision.model is not None:
        speaking.append(Witness.MODEL)
    if decision.syntax:
        speaking.append(Witness.SYNTAX)
    return tuple(speaking)


def provenance_of(decision: Decision, profile: WitnessProfile) -> str:
    """What to persist about how this field was settled.

    Not the fusion outcome alone. `AGREE` is what `fuse` concluded, and it
    concludes that whether syntax supported the reading or was never asked —
    which is correct for a decision and wrong for a record. The stored plan has
    to say which readers were consulted, because that is what a later reader
    needs and what fusion's own vocabulary cannot express.
    """
    spoke = witnesses_of(decision)

    if decision.outcome is not Fusion.AGREE:
        return decision.outcome.value

    if len(spoke) > 1:
        return "AGREE"
    if Witness.MODEL in spoke:
        return "MODEL_ONLY_ACCEPTED"
    return "SYNTAX_ONLY_ACCEPTED"


@dataclass(frozen=True)
class SettledField:
    """One field, its value, and honestly how it came to be settled."""

    field: str
    value: object
    provenance: str
    witnesses: Sequence[str]
    reader_id: str = ""
    detail: str = ""

    def to_json(self) -> dict:
        return {"field": self.field, "value": self.value,
                "provenance": self.provenance,
                "witnesses": list(self.witnesses),
                "reader_id": self.reader_id, "detail": self.detail}


def record(decisions: Sequence[Decision], profile: WitnessProfile,
           ) -> Sequence[SettledField]:
    """Every decision, in the shape a plan should store it."""
    return tuple(
        SettledField(
            field=d.dimension,
            value=d.value if d.proceeds else None,
            provenance=provenance_of(d, profile),
            witnesses=[w.value for w in witnesses_of(d)],
            reader_id="" if d.model is None else d.model.reader_id,
            detail=d.detail)
        for d in decisions)
