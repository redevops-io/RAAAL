"""Visual semantics registry — one declaration per relation type.

Without this, every component maps relations independently and the same edge is
called "invalidates" on one page and "affects" on another. The typed relation
*is* the information; degrading it to "related to" throws away the thing the
ontology exists to preserve.

The registry owns **language and rendering semantics**. It deliberately does not
own layout — geometry belongs to `GraphViewModel.layout()`, so a component may
choose where to draw an edge but never what it means.

Every consumer reads from here: impact graph, relation badge, version timeline,
library headlines, accessibility output, and any future Discovery proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping

from .chain import Adversity


class EdgeStyle(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    """Reserved for *directness*, never for effect — the channels stay orthogonal."""


class Direction(str, Enum):
    OUTBOUND = "outbound"
    """From the subject to something it affects."""

    INBOUND = "inbound"
    """From a contributor toward the subject."""


@dataclass(frozen=True)
class RelationSemantics:
    """How one relation type reads, everywhere it appears."""

    relation: str
    label: str
    """Plain-language form. Used in badges, tables and prose alike."""

    effect: Adversity
    direction: Direction
    summary_template: str
    """One sentence with `{source}` and `{target}` placeholders."""

    edge_style: EdgeStyle = EdgeStyle.SOLID
    note: str = ""
    """Why the effect is what it is, where that is not obvious."""

    def summarize(self, source: str, target: str) -> str:
        return self.summary_template.format(source=source, target=target)

    def aria(self, source: str, target: str) -> str:
        return f"{source} {self.label} {target}; effect {self.effect.value}"

    def to_json(self) -> Dict[str, object]:
        return {
            "relation": self.relation,
            "label": self.label,
            "effect": self.effect.value,
            "direction": self.direction.value,
            "edge_style": self.edge_style.value,
            "note": self.note,
        }


def _r(relation, label, effect, direction, template, note="") -> RelationSemantics:
    return RelationSemantics(
        relation=relation, label=label, effect=effect, direction=direction,
        summary_template=template, note=note,
    )


#: The single declaration. Adding a relation type without an entry is an error
#: at render time, not a silent default — see `resolve`.
RELATION_SEMANTICS: Mapping[str, RelationSemantics] = {
    s.relation: s for s in (
        _r("INVALIDATES_RESULTS_OF", "invalidates results of", Adversity.BLOCKING,
           Direction.OUTBOUND, "Invalidates the published results of {target}."),
        _r("REFUTES", "refutes", Adversity.BLOCKING, Direction.OUTBOUND,
           "Refutes {target}; contradicting evidence prevails."),
        _r("QUALIFIES", "qualifies", Adversity.ADVISORY, Direction.OUTBOUND,
           "Narrows where {target} holds without overturning it.",
           note="Narrowing scope is not refutation and must not render as one."),
        _r("CORRECTED", "corrected", Adversity.AFFECTING, Direction.OUTBOUND,
           "Produced a published correction to {target}."),
        _r("MOTIVATED", "motivated", Adversity.NONE, Direction.OUTBOUND,
           "Caused {target} to be created.",
           note="Constructive. A renderer treating every finding edge as adverse "
                "would assert the finding damaged the thing it created."),
        _r("INTRODUCED", "introduced", Adversity.NONE, Direction.OUTBOUND,
           "Forced {target} to be declared."),
        _r("SUPERSEDED_BY", "superseded by", Adversity.AFFECTING, Direction.OUTBOUND,
           "Replaced by {target}; retained and still reachable."),
        _r("SUPPORTS", "supports", Adversity.NONE, Direction.INBOUND,
           "{source} supports {target}."),
        _r("CONTRADICTS", "contradicts", Adversity.NONE, Direction.INBOUND,
           "{source} contradicts the claim {target} synthesises.",
           note="An input edge. Evidence contradicting a claim is not adverse to "
                "the finding that records the contradiction — the adverse effect "
                "occurs downstream, on the claim."),
        _r("DEPENDS_ON", "depends on", Adversity.NONE, Direction.INBOUND,
           "{source} rests on {target}.",
           note="Depending on an assumption is not itself adverse; whether the "
                "assumption is validated is a separate, node-level fact."),
        _r("REALIZED_BY", "realized by", Adversity.NONE, Direction.OUTBOUND,
           "{source} is enforced by {target}."),
    )
}


class UndeclaredRelation(KeyError):
    """Raised when a relation type has no visual semantics.

    A fallback would make the interface silently optimistic, which is the worst
    available default for a provenance system: the relations most likely to be
    missing are the new ones, and a new relation is exactly the case where
    assuming "harmless" is least safe.
    """


def resolve(relation: str) -> RelationSemantics:
    """Look up a relation's semantics, or refuse to render it."""
    if relation not in RELATION_SEMANTICS:
        raise UndeclaredRelation(
            f"relation {relation!r} has no visual semantics. Declare it in "
            "RELATION_SEMANTICS — every relation must state its effect and "
            "language before it can render anywhere."
        )
    return RELATION_SEMANTICS[relation]


def effect_of(relation: str) -> Adversity:
    """The single source for a relation's consequence."""
    return resolve(relation).effect


def label_of(relation: str) -> str:
    return resolve(relation).label
