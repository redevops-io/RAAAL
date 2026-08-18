"""What a reader proposed, and what fusion concluded — the shapes, not the rules.

Carriers. `Proposal` is one reader's answer for one dimension and `Decision` is
the conclusion about it; neither decides anything. They live apart from the
fusion that produced them because that fusion is gone: comparison, aggregation
and the outcome itself belong to `discovery-runtime` now, and these are the
records Quantify passes around them.

Kept in Quantify deliberately. A `Proposal` carries `reader_id` and
`source_span` because a stored plan has to say who read what and where, and a
`Decision` carries `material` because whether a dimension matters is a domain
judgement. Those are application facts about how Quantify records a reading,
not an alternative implementation of Discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from discovery_runtime.fusion import Fusion

from .binding import RelationBinding
from .reader import ReadingSet
from .semantics import PIPELINE_VERSION, SemanticCandidate
from .syntax import SyntaxEvidence, Value

#: Which fusion policy a stored decision was made under.
POLICY_VERSION = "quantify-fusion@1"


@dataclass(frozen=True)
class Proposal:
    """One reader's answer for one dimension, whoever the reader is."""

    dimension: str
    value: Any
    reader_id: str
    source_span: str = ""


@dataclass(frozen=True)
class Decision:
    """What fusion concluded, and everything needed to explain it later."""

    dimension: str
    outcome: Fusion
    value: Any = None
    """Present only when the outcome proceeds. Deliberately `None` otherwise:
    a decision that carried a value alongside `DISAGREE` is a value a caller
    will render, and then a figure exists for a question that was not settled.
    """

    material: bool = True
    model: Optional[Proposal] = None
    syntax: Sequence[SyntaxEvidence] = ()
    detail: str = ""
    policy_version: str = POLICY_VERSION

    @property
    def proceeds(self) -> bool:
        return self.outcome.proceeds

    def to_json(self) -> dict:
        return {"dimension": self.dimension, "outcome": self.outcome.value,
                "value": self.value, "material": self.material,
                "model": (None if self.model is None else
                          {"value": self.model.value,
                           "reader_id": self.model.reader_id,
                           "source_span": self.model.source_span}),
                "syntax": [e.to_json() for e in self.syntax],
                "detail": self.detail, "policy_version": self.policy_version}




#: Dimensions where a trailing `m` counts periods rather than millions.
#:


@dataclass(frozen=True)
class Read:
    """Everything both witnesses said about one utterance, and what was decided."""

    text: str
    values: Sequence[Value] = ()
    bindings: Sequence[RelationBinding] = ()
    candidates: Sequence[SemanticCandidate] = ()
    model: Optional[ReadingSet] = None
    decisions: Sequence[Decision] = ()
    pipeline_version: str = PIPELINE_VERSION

    @property
    def by_field(self) -> Mapping[str, Decision]:
        return {d.dimension: d for d in self.decisions}

    @property
    def settled(self) -> Mapping[str, Any]:
        return {d.dimension: d.value for d in self.decisions if d.proceeds}

    @property
    def open(self) -> Sequence[Decision]:
        return tuple(d for d in self.decisions if not d.proceeds)

    @property
    def intermediate(self) -> Sequence[SemanticCandidate]:
        """Semantics this pipeline computed that the contract does not carry.

        Kept and reported rather than dropped: `amount_kind=fixed` is a real
        reading of a real sentence, and discarding it because no contract field
        exists would lose the evidence that the contract might need one."""
        return tuple(c for c in self.candidates if not c.is_contract_field)
