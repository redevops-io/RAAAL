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

MAPPER_VERSION = "quantify-semantics@2"

#: **Contract field names are canonical at the fusion boundary.**
#:
#: Readers, mappers, fusion and corpus assertions all speak these. A parser
#: feature may be called whatever is clearest locally — `rebalancing_cadence`
#: reads better inside this file than `periodic_rebalancing` does — but it must
#: be translated before it becomes a `SemanticCandidate`, because the moment two
#: witnesses name the same thing differently they can never agree about it.
#:
#: That is not hypothetical: `rebalancing_cadence` produced a DISAGREE with the
#: model silent, purely because the schema calls it `periodic_rebalancing`. A
#: vocabulary duplication reported as a reading conflict.
def contract_fields() -> frozenset:
    from .schema import QUANTIFY_SCHEMA

    return frozenset(d.name for d in QUANTIFY_SCHEMA.dimensions)


#: Semantics this pipeline computes that the contract does not carry.
#:
#: Adjudicated one at a time rather than by growing the schema to match the
#: mapper. A field belongs here when the *runtime* has no use for it — when
#: promoting it would add a dimension nothing executes, which is the
#: declared-but-not-executed shape this project already has a manifest to
#: prevent.
#:
#:     amount_kind          fixed | proportional | residual. Derivable metadata
#:                          around `amount`; the manifest executes an amount,
#:                          not a kind of amount.
#:     holding_period_days  an intermediate reading. Nothing in the manifest
#:                          makes a holding period change a result.
#:     account_allocation   which account holds which split. Real, and a
#:                          *relation* rather than a dimension — the schema
#:                          already models this shape with `portfolio_sleeves`
#:                          and `account_transition`, and a flat field would be
#:                          the flattening `RelationSpec` exists to refuse.
#:
#: These still produce candidates, and the candidates still carry evidence.
#: What they do not do is enter fusion as contract fields, because there is no
#: contract field for the other witness to answer with.
#:     rebalancing_cadence  the rename to `periodic_rebalancing` was correct in
#:                          *name* and wrong in *shape*. That dimension holds a
#:                          free-text description — its own examples are
#:                          "rebalance quarterly" and "when it drifts more than
#:                          5 points" — and the deterministic path produces the
#:                          canonical token `annual`. Under the dimension's
#:                          declared TEXT comparison those are not the same
#:                          value, so the rename turned a vocabulary mismatch
#:                          into a permanent false DISAGREE.
#:
#:                          Left intermediate rather than resolved either way.
#:                          Making the mapper emit prose to match a reader is
#:                          worse than the mismatch, and giving the dimension a
#:                          looser comparison policy is a schema decision.
INTERMEDIATE_FIELDS = frozenset({
    "amount_kind", "holding_period_days", "account_allocation",
    "rebalancing_cadence"})

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
    """A contract field name, or one of `INTERMEDIATE_FIELDS`. Never a third
    thing — `test_semantics.py` asserts it, because a candidate naming a field
    neither the contract nor the intermediate list knows is a candidate no
    other witness can ever answer."""

    value: Any
    source_value_id: str
    binding_id: str
    source_span: str = ""
    """The characters this candidate rests on — the value's own span, not the
    sentence around it.

    Carried because fusion's ambiguity check reads it, and a caller that handed
    over the whole utterance made *every* field of a sentence containing
    "rebalance" ambiguous. The day rule in "rebalance on the last session of
    each quarter" is perfectly determinate; what is ambiguous is what
    rebalancing does, which is a different dimension of the same sentence."""

    evidence: Sequence[str] = ()
    mapper_version: str = MAPPER_VERSION

    @property
    def is_contract_field(self) -> bool:
        return self.field not in INTERMEDIATE_FIELDS

    def to_json(self) -> dict:
        return {"field": self.field, "value": self.value,
                "is_contract_field": self.is_contract_field,
                "source_value_id": self.source_value_id,
                "binding_id": self.binding_id,
                "source_span": self.source_span,
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
    FieldMapping(field="stated_weights", relation="shares_head_with",
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


@dataclass(frozen=True)
class DerivationFamily:
    """A field whose value comes from the *evidence*, not from the literal.

    `contribute a fixed $500` carries the amount in the literal and the
    *kind* of amount in the word `fixed`. A family says which evidence
    determines which value, and it fires only when that evidence is present —
    never as a default, and never on partial support.

    The constraint that shapes all of these: a family may propose a value only
    when the linguistic evidence genuinely determines it. `below` alone does
    not mean a persistent condition; `is below` does, and `crosses below` means
    the opposite. So the discriminator is the verb, and the preposition is only
    the signal that a comparison is happening at all. A family requiring one
    without the other would be inferring from a preposition, which is how a
    mapper starts guessing.
    """

    name: str
    field: str
    value: Any
    value_kinds: frozenset
    #: Every one of these must be among the binding's modifiers.
    needs_modifiers: frozenset = frozenset()
    #: At least one of these must be.
    needs_any_modifier: frozenset = frozenset()
    #: The bound target's lemma must be one of these.
    needs_target: frozenset = frozenset()
    #: None of these may be among the modifiers.
    forbids_modifiers: frozenset = frozenset()


#: Verbs that describe a *change* of state — the crossing itself.
_DYNAMIC = frozenset({"cross", "drop", "fall", "rise", "break", "go", "move",
                      "dip", "climb", "decline"})

#: Prepositions that mark a comparison against a level. Necessary and never
#: sufficient: what separates an event from a state is the verb beside them.
_COMPARISON = frozenset({"below", "under", "above", "beneath", "over"})

DERIVATIONS: Sequence[DerivationFamily] = (
    # amount_kind — the literal says how much, a modifier says what sort
    DerivationFamily(
        name="fixed amount", field="amount_kind", value="fixed",
        value_kinds=frozenset({"money"}),
        needs_any_modifier=frozenset({"fix", "fixed", "flat", "set"})),
    DerivationFamily(
        name="proportional amount", field="amount_kind", value="proportional",
        value_kinds=frozenset({"percentage"}),
        needs_any_modifier=frozenset({"of"})),

    # trigger_semantics — decided by the verb, signalled by the preposition
    DerivationFamily(
        name="crossing event", field="trigger_semantics", value="crossing_event",
        value_kinds=frozenset({"duration", "moving_average_window",
                               "percentage"}),
        needs_any_modifier=_COMPARISON, needs_target=_DYNAMIC),
    DerivationFamily(
        name="persistent condition", field="trigger_semantics",
        value="persistent_condition",
        value_kinds=frozenset({"duration", "moving_average_window"}),
        needs_any_modifier=_COMPARISON, needs_modifiers=frozenset({"be"}),
        forbids_modifiers=_DYNAMIC),

    # day_rule — an ordinal on the period noun
    DerivationFamily(
        name="last session of period", field="day_rule",
        value="last_session_of_period", value_kinds=frozenset({"cadence"}),
        needs_any_modifier=frozenset({"last", "final"})),
    DerivationFamily(
        name="first session of period", field="day_rule",
        value="first_session_of_period", value_kinds=frozenset({"cadence"}),
        needs_any_modifier=frozenset({"first"})),
)


def derive(bindings: Sequence[RelationBinding], values: Sequence[Value],
           families: Sequence[DerivationFamily] = DERIVATIONS,
           ) -> Sequence[SemanticCandidate]:
    """Candidates whose value comes from evidence rather than from the literal."""
    by_id = {value_id(value): value for value in values}
    candidates = []

    for binding in bindings:
        if not binding.established:
            continue
        value = by_id.get(binding.value_id)
        if value is None:
            continue
        modifiers = set(binding.modifiers)
        for family in families:
            if value.kind not in family.value_kinds:
                continue
            if not family.needs_modifiers <= modifiers:
                continue
            if family.needs_any_modifier and not (
                    family.needs_any_modifier & modifiers):
                continue
            if family.needs_target and binding.target_lemma not in family.needs_target:
                continue
            if family.forbids_modifiers & modifiers:
                continue
            if family.forbids_modifiers and binding.target_lemma in family.forbids_modifiers:
                continue
            candidates.append(SemanticCandidate(
                field=family.field, value=family.value,
                source_value_id=binding.value_id,
                binding_id=binding_id(binding),
                source_span=value.source_span,
                evidence=tuple(binding.evidence) + (
                    f"family={family.name}",
                    f"target={binding.target_lemma}",
                    f"modifiers={sorted(modifiers)}")))
    return tuple(candidates)


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
            # Matched on the *lemma*, never the rebuilt span. `put in $1,000
            # each quarter` binds to a target whose readable span is "put in"
            # and whose lemma is `put`, and a table compared against the span
            # missed exactly the multiword cases.
            if (mapping.target_in is not None
                    and binding.target_lemma not in mapping.target_in):
                continue
            candidates.append(SemanticCandidate(
                field=mapping.field,
                value=_value_for(mapping, value, binding),
                source_value_id=binding.value_id,
                binding_id=binding_id(binding),
                source_span=value.source_span,
                evidence=tuple(binding.evidence) + (
                    f"target={binding.target_span}",)))
    return tuple(candidates) + derive(bindings, values)


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
