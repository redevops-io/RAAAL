"""A runtime decision: what was decided, on what evidence, and why.

Not a span. A span is an interval — it answers *when* and *how long*. A decision
is a choice the runtime made on the user's behalf, and the questions that matter
later are all of that second kind:

    Why wasn't this benchmark included?
    Why did this become AFTER_RESULTS?
    Why did we ask another question instead of running it?

Recorded as its own object rather than as attributes on whichever span happened
to be open, because a decision buried in a span is only findable by someone who
already knows which span to open.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class DecisionKind(str, Enum):
    """Where in the chain a choice was made. One per point at which the runtime
    could have done otherwise."""

    INPUT = "INPUT"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    QUARANTINE = "QUARANTINE"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    PROPOSAL_GENERATION = "PROPOSAL_GENERATION"
    CONFIRMATION = "CONFIRMATION"
    RUN_SELECTION = "RUN_SELECTION"
    BENCHMARK_SELECTION = "BENCHMARK_SELECTION"
    PUBLICATION_GATE = "PUBLICATION_GATE"


@dataclass(frozen=True)
class Decision:
    """One choice, with what it rests on."""

    decision_id: str
    trace_id: str
    kind: DecisionKind
    outcome: str
    """What was decided, in the runtime's own vocabulary — `AFTER_RESULTS`,
    `REFUSED`, `UNCLASSIFIED`. Not prose: this is the field a dashboard groups
    by, and prose does not group."""

    reason: str
    """Why, in a sentence a person can read."""

    evidence_refs: Sequence[str] = ()
    """What the decision rested on, by reference: rule ids, artifact ids,
    ruleset refs, the hash of the instruction. References rather than content,
    for the same reason the artifact graph holds references — evidence copied
    into a log is evidence that can disagree with its source."""

    confidence: Optional[float] = None
    """`None` where the notion does not apply.

    A deterministic rule match has no confidence; it has a rule. Writing 1.0
    would manufacture a number, and a column of 1.0s teaches a reader that
    confidence is meaningful here when the only real values will come from
    somewhere else entirely. Populated solely when a source actually reports
    one."""

    alternatives_considered: Sequence[str] = ()
    """What else was available and not chosen. The half of "why" that a bare
    outcome omits: "benchmark X was included" is far less useful than the same
    line beside the three that were not."""

    at: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"decision_id": self.decision_id, "trace_id": self.trace_id,
                "kind": self.kind.value, "outcome": self.outcome,
                "reason": self.reason,
                "evidence_refs": list(self.evidence_refs),
                "confidence": self.confidence,
                "alternatives_considered": list(self.alternatives_considered),
                "at": self.at}


def new_decision_id() -> str:
    return f"dec-{uuid.uuid4().hex[:16]}"
