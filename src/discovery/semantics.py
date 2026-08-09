"""Which field does this binding fill? — typed candidates, decided by fusion.

The last step before the contract, and the narrowest one. It consumes
`RelationBinding` and normalised values, and emits `SemanticCandidate`. It never
touches a parse: the binder already turned structure into a typed fact, and a
mapper that went back to the tokens would be the third place in this pipeline
reading dependencies.

    normalisation → binding → semantic candidate → fusion → field | Unresolved

**A candidate is a proposal, not a field.** Nothing here decides that a
candidate survives; that is fusion's job, and the separation is what stops this
module from quietly becoming the authority. So a candidate carries no
confidence and no rank — only what it proposes, what it came from, and the
evidence that produced it.

**Mappings are added when a pending case demands one, and not before.** A
generic semantic ontology written up front would be an ontology tuned on
nothing; the five here are the pairings the binder already declares itself
evidence for, in the order the corpus needs them. What a mapping cannot do is
invent a producer: a case like *"weight by inverse volatility"* carries no
normalised value at all, so no binding exists and no candidate can be proposed.
That case is not unmapped — it is a job for the semantic reader, and the
closure report says so by name rather than leaving it in a pile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .binding import RelationBinding, value_id
from .syntax import Value

MAPPER_VERSION = "quantify-semantics@1"

#: Verbs that mean "money arrives", against those that mean something else
#: happens on a schedule. This is the smallest table that separates the two
#: cadences in "contribute $500 monthly, rebalanced annually", and it is a
#: *mapping* decision rather than a scoring one — the binder already said which
#: verb each cadence attaches to, and this says what that verb means here.
FUNDING_VERBS = frozenset({
    "contribute", "invest", "deposit", "add", "save", "buy", "put",
    "purchase", "transfer", "fund", "max"})

REBALANCING_VERBS = frozenset({
    "rebalance", "reallocate", "adjust", "harvest", "review", "trim"})


@dataclass(frozen=True)
class SemanticCandidate:
    """One proposed field, and everything it was derived from."""

    field: str
    value: Any
    source_value_id: str
    binding_id: str
    evidence: Sequence[str] = ()
    mapper_version: str = MAPPER_VERSION

    def to_json(self) -> dict:
        return {"field": self.field, "value": self.value,
                "source_value_id": self.source_value_id,
                "binding_id": self.binding_id,
                "evidence": list(self.evidence),
                "mapper_version": self.mapper_version}


def binding_id(binding: RelationBinding) -> str:
    """A stable name for one binding, derived the same way a value's is."""
    return f"{binding.value_id}:{binding.relation}"


@dataclass(frozen=True)
class FieldMapping:
    """One structural relation, one field, and what it needs to fire."""

    field: str
    relation: str
    value_kinds: frozenset
    pairing: str
    """The binder's own `supports` label, matched so a mapping cannot claim a
    relation the binder never offered it as evidence for."""

    target_in: Optional[frozenset] = None
    """When set, the binding's target must be one of these lemmas. Used by the
    cadence mappings, where the same relation fills a different field depending
    on which verb the value attaches to."""


MAPPINGS: Sequence[FieldMapping] = (
    # 1. asset ↔ weight
    FieldMapping(field="asset_weight", relation="shares_head_with",
                 value_kinds=frozenset({"percentage", "ratio"}),
                 pairing="asset↔weight"),
    # 2. account ↔ allocation
    FieldMapping(field="account_allocation", relation="appositive_of",
                 value_kinds=frozenset({"ratio", "percentage"}),
                 pairing="account↔allocation"),
    # 3. cadence ↔ action, split by what the action is
    FieldMapping(field="cadence", relation="governed_by",
                 value_kinds=frozenset({"cadence"}),
                 pairing="cadence↔action", target_in=FUNDING_VERBS),
    FieldMapping(field="rebalancing_cadence", relation="governed_by",
                 value_kinds=frozenset({"cadence"}),
                 pairing="cadence↔action", target_in=REBALANCING_VERBS),
    # 4. condition ↔ action
    FieldMapping(field="moving_average_window", relation="governed_by",
                 value_kinds=frozenset({"moving_average_window"}),
                 pairing="condition↔action"),
    # 5. timing ↔ action
    FieldMapping(field="holding_period_days", relation="governed_by",
                 value_kinds=frozenset({"duration"}),
                 pairing="timing↔action"),
    FieldMapping(field="amount", relation="governed_by",
                 value_kinds=frozenset({"money"}),
                 pairing="cadence↔action", target_in=FUNDING_VERBS),
)


class Unmappable(str):
    """Why no candidate could be proposed. A string subclass so it reads as the
    reason it is, and never as a value."""


def propose(bindings: Sequence[RelationBinding], values: Sequence[Value],
            mappings: Sequence[FieldMapping] = MAPPINGS,
            ) -> Sequence[SemanticCandidate]:
    """Every field candidate the declared mappings support.

    An unestablished binding proposes nothing. That is the whole reason
    `INSUFFICIENT_RELATION` exists downstream — a candidate built on an
    `AMBIGUOUS` binding would be a coin toss with a field name on it.
    """
    by_id = {value_id(value): value for value in values}
    candidates = []

    for binding in bindings:
        if not binding.established:
            continue
        value = by_id.get(binding.value_id)
        if value is None:
            continue
        for mapping in mappings:
            if mapping.relation != binding.relation:
                continue
            if value.kind not in mapping.value_kinds:
                continue
            if mapping.pairing not in binding.supports:
                continue
            if mapping.target_in is not None and (
                    binding.target_span.lower().rstrip("d")
                    not in {v.rstrip("d") for v in mapping.target_in}
                    and binding.target_span.lower() not in mapping.target_in):
                continue
            candidates.append(SemanticCandidate(
                field=mapping.field,
                value=_value_for(mapping, value, binding),
                source_value_id=binding.value_id,
                binding_id=binding_id(binding),
                evidence=tuple(binding.evidence) + (
                    f"target={binding.target_span}",)))
    return tuple(candidates)


def _value_for(mapping: FieldMapping, value: Value,
               binding: RelationBinding) -> Any:
    """What the candidate proposes.

    For the two pairings whose whole content is a relationship — a weight and
    its asset, an allocation and its account — the value is the *pair*.
    Flattening it to the number would throw away the binding that made the
    candidate possible, which is the failure `RelationSpec` exists to name.
    """
    if mapping.field in ("asset_weight", "account_allocation"):
        return {"target": binding.target_span, "value": value.canonical}
    return value.canonical


def by_field(candidates: Sequence[SemanticCandidate]) -> Mapping[str, list]:
    grouped: dict = {}
    for candidate in candidates:
        grouped.setdefault(candidate.field, []).append(candidate)
    return grouped
