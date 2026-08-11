"""Governed three-way merge for methodologies.

Structural merge tooling answers "do these two edits combine?". It cannot answer
"is the combination a coherent strategy?" — and in this domain the second
question is the one that matters. A merge can be textually clean, schema-valid,
and still produce a portfolio that breaks its leverage cap or selects no assets
at all.

So merge runs in layers and reports a verdict per layer rather than a boolean::

    Syntactic      do the trees combine?
        ↓
    Contract       does the result still satisfy the output contract?
        ↓
    Economic       is the result a coherent strategy?
        ↓
    Comparability  do previously published results still mean anything?

`comparability_status` is the one with no prior art, and it is what makes
supersession computable: it answers whether the number published for v2 can
still be cited now that v3 exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spec import FIELD_SEMANTICS, Methodology, Semantics

# Weights summing outside 1 +/- this are treated as un-normalized. Loose enough
# to absorb float noise from independent edits, tight enough to catch a genuine
# mistake.
NORMALIZATION_TOLERANCE = 1e-6

# A single asset above this share of gross exposure is flagged. Not a hard rule
# — it is a review trigger, because concentration that arrives via merge was
# chosen by nobody.
CONCENTRATION_THRESHOLD = 0.50


class StructuralStatus(str, Enum):
    CLEAN = "clean"
    CONFLICTED = "conflicted"
    FAILED = "failed"


class ContractStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"


class EconomicStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


class ComparabilityStatus(str, Enum):
    COMPARABLE = "comparable"
    BROKEN = "broken"


@dataclass(frozen=True)
class Conflict:
    field: str
    semantics: Semantics
    detail: str
    ours: Any = None
    theirs: Any = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "semantics": self.semantics.value,
            "detail": self.detail,
            "ours": self.ours,
            "theirs": self.theirs,
        }


@dataclass
class MergeResult:
    """The verdict. Deliberately not a boolean."""

    structural_status: StructuralStatus
    contract_status: ContractStatus
    economic_status: EconomicStatus
    comparability_status: ComparabilityStatus
    merged: Optional[Methodology] = None
    unresolved_conflicts: List[Conflict] = field(default_factory=list)
    required_retests: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        """A merge may only be published if every layer passes.

        `NEEDS_REVIEW` deliberately blocks: an economic anomaly that nobody has
        looked at is not a publishable strategy.
        """
        return (
            self.structural_status is StructuralStatus.CLEAN
            and self.contract_status is ContractStatus.SATISFIED
            and self.economic_status is EconomicStatus.VALID
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "structural_status": self.structural_status.value,
            "contract_status": self.contract_status.value,
            "economic_status": self.economic_status.value,
            "comparability_status": self.comparability_status.value,
            "publishable": self.publishable,
            "unresolved_conflicts": [c.to_json() for c in self.unresolved_conflicts],
            "required_retests": list(self.required_retests),
            "notes": list(self.notes),
            "merged_version_id": self.merged.version_id if self.merged else None,
            "merged_content_hash": self.merged.content_hash if self.merged else None,
        }


# --- layer 1: syntactic ----------------------------------------------------


def _merge_unordered(base: Sequence, ours: Sequence, theirs: Sequence, key=lambda x: x):
    """Set union with base-aware deletion.

    Additions from either side are kept; a deletion on one side wins unless the
    other side modified the same element, which is a conflict.
    """
    b = {key(x): x for x in base}
    o = {key(x): x for x in ours}
    t = {key(x): x for x in theirs}

    merged: Dict[Any, Any] = {}
    conflicts: List[Tuple[Any, Any, Any]] = []

    for k in set(b) | set(o) | set(t):
        in_b, in_o, in_t = k in b, k in o, k in t
        if in_o and in_t:
            if o[k] == t[k]:
                merged[k] = o[k]
            elif in_b and o[k] == b[k]:
                merged[k] = t[k]           # only theirs changed it
            elif in_b and t[k] == b[k]:
                merged[k] = o[k]           # only ours changed it
            else:
                conflicts.append((k, o[k], t[k]))
        elif in_o and not in_t:
            if in_b:
                continue                   # theirs deleted; ours left it alone
            merged[k] = o[k]               # ours added
        elif in_t and not in_o:
            if in_b:
                continue                   # ours deleted
            merged[k] = t[k]               # theirs added
    return merged, conflicts


def _merge_ordered(base: Sequence, ours: Sequence, theirs: Sequence):
    """Order carries meaning, so only one side may reorder.

    No attempt is made to interleave two independent reorderings — for a
    rebalance pipeline that would silently invent an execution order neither
    author wrote.
    """
    if list(ours) == list(theirs):
        return list(ours), None
    if list(ours) == list(base):
        return list(theirs), None
    if list(theirs) == list(base):
        return list(ours), None
    return None, (list(ours), list(theirs))


def _merge_scalar_map(base: Dict, ours: Dict, theirs: Dict):
    """Per-key three-way merge for parameter maps."""
    merged: Dict[str, Any] = {}
    conflicts: List[Tuple[str, Any, Any]] = []
    for k in set(base) | set(ours) | set(theirs):
        b, o, t = base.get(k), ours.get(k), theirs.get(k)
        if o == t:
            if o is not None:
                merged[k] = o
        elif o == b:
            if t is not None:
                merged[k] = t
        elif t == b:
            if o is not None:
                merged[k] = o
        else:
            conflicts.append((k, o, t))
    return merged, conflicts


# --- layer 3: economic -----------------------------------------------------


def _check_economics(m: Methodology) -> Tuple[EconomicStatus, List[str], List[str]]:
    """Is the merged artifact a coherent strategy?

    These are the failures that pass layers 1 and 2 and would otherwise reach
    publication: normalization broken by two independent weight edits, a leverage
    cap exceeded by their sum, an empty universe from ANDed filters, and
    concentration nobody chose.
    """
    problems: List[str] = []
    retests: List[str] = []
    status = EconomicStatus.VALID

    if m.scoring_terms:
        total = sum(m.scoring_terms.values())
        if abs(total) < NORMALIZATION_TOLERANCE:
            problems.append("scoring terms sum to zero — the score is undefined")
            status = EconomicStatus.INVALID
        retests.append("re-run backtest: scoring weights changed")

    weights = {k: v.value for k, v in m.params.items() if k.startswith("weight_")}
    if weights:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
            problems.append(
                f"weights sum to {total:.6f}, not 1.0 — merged edits broke normalization"
            )
            status = EconomicStatus.INVALID
        gross = sum(abs(float(v)) for v in weights.values())
        if gross > m.contract.gross_leverage_max + NORMALIZATION_TOLERANCE:
            problems.append(
                f"gross leverage {gross:.4f} exceeds contract max "
                f"{m.contract.gross_leverage_max}"
            )
            status = EconomicStatus.INVALID
        for name, value in weights.items():
            if gross > 0 and abs(float(value)) / gross > CONCENTRATION_THRESHOLD:
                problems.append(
                    f"{name} is {abs(float(value)) / gross:.0%} of gross exposure — "
                    "concentration arrived via merge, not by decision"
                )
                if status is EconomicStatus.VALID:
                    status = EconomicStatus.NEEDS_REVIEW

    # Conjunction sets can intersect to nothing. Cheap to check, and an empty
    # universe is otherwise discovered only at run time.
    remaining = [a for a in m.contract.universe if a not in set(m.excluded_assets)]
    if not remaining:
        problems.append(
            "universe is empty after exclusions — merged filters select no assets"
        )
        status = EconomicStatus.INVALID

    if m.universe_filters:
        retests.append("re-run universe selection: filters changed")

    return status, problems, retests


# --- layer 4: comparability ------------------------------------------------


def _check_comparability(
    base: Methodology, merged: Methodology
) -> Tuple[ComparabilityStatus, List[str]]:
    """Can results published against `base` still be cited?

    Breaks when the thing being measured changed — a different universe, a
    different rebalance cadence, or a different cost model all mean the old
    number describes a different strategy, even though the concept id is
    unchanged.
    """
    notes: List[str] = []
    contract_breaks = merged.contract.breaks_compatibility_with(base.contract)
    if contract_breaks:
        notes.extend(contract_breaks)
    if merged.cost_model != base.cost_model:
        notes.append(f"cost model {base.cost_model} -> {merged.cost_model}")
    if set(merged.excluded_assets) != set(base.excluded_assets):
        notes.append("exclusion set changed — prior results covered a different universe")

    if notes:
        return ComparabilityStatus.BROKEN, notes
    return ComparabilityStatus.COMPARABLE, []


# --- entry point -----------------------------------------------------------


def merge(base: Methodology, ours: Methodology, theirs: Methodology) -> MergeResult:
    """Three-way merge two independent revisions of one methodology.

    All three MUST share a concept id — merging different methods is not a merge,
    it is a category error, and silently producing something from it would be
    worse than failing.
    """
    if not (base.concept == ours.concept == theirs.concept):
        return MergeResult(
            structural_status=StructuralStatus.FAILED,
            contract_status=ContractStatus.VIOLATED,
            economic_status=EconomicStatus.INVALID,
            comparability_status=ComparabilityStatus.BROKEN,
            notes=[
                f"cannot merge different concepts: {base.concept}, "
                f"{ours.concept}, {theirs.concept}"
            ],
        )

    conflicts: List[Conflict] = []
    notes: List[str] = []

    # rules — unordered set, keyed by id
    merged_rules, rule_conflicts = _merge_unordered(
        base.rules, ours.rules, theirs.rules, key=lambda r: r.id
    )
    for rid, o, t in rule_conflicts:
        conflicts.append(
            Conflict(
                field="rules",
                semantics=Semantics.UNORDERED_SET,
                detail=f"rule {rid!r} modified differently on both sides",
                ours=o.to_json(),
                theirs=t.to_json(),
            )
        )

    # pipeline / fallback_chain — ordered
    merged_pipeline, pipe_conflict = _merge_ordered(base.pipeline, ours.pipeline, theirs.pipeline)
    if pipe_conflict:
        conflicts.append(
            Conflict(
                field="pipeline",
                semantics=Semantics.ORDERED_SEQUENCE,
                detail="both sides reordered the execution pipeline; order carries meaning",
                ours=pipe_conflict[0],
                theirs=pipe_conflict[1],
            )
        )
        merged_pipeline = list(base.pipeline)

    merged_fallback, fb_conflict = _merge_ordered(
        base.fallback_chain, ours.fallback_chain, theirs.fallback_chain
    )
    if fb_conflict:
        conflicts.append(
            Conflict(
                field="fallback_chain",
                semantics=Semantics.ORDERED_SEQUENCE,
                detail="both sides reordered the fallback chain; precedence carries meaning",
                ours=fb_conflict[0],
                theirs=fb_conflict[1],
            )
        )
        merged_fallback = list(base.fallback_chain)

    # params — scalar map
    merged_params, param_conflicts = _merge_scalar_map(base.params, ours.params, theirs.params)
    for key, o, t in param_conflicts:
        conflicts.append(
            Conflict(
                field="params",
                semantics=Semantics.SCALAR,
                detail=f"parameter {key!r} set to different values on both sides",
                ours=o.to_json() if o else None,
                theirs=t.to_json() if t else None,
            )
        )

    # scoring_terms — weighted expression; relative magnitudes matter, so a
    # differing edit to the same term is a conflict rather than a pick.
    merged_scores, score_conflicts = _merge_scalar_map(
        base.scoring_terms, ours.scoring_terms, theirs.scoring_terms
    )
    for key, o, t in score_conflicts:
        conflicts.append(
            Conflict(
                field="scoring_terms",
                semantics=Semantics.WEIGHTED_EXPRESSION,
                detail=f"scoring term {key!r} reweighted differently on both sides",
                ours=o,
                theirs=t,
            )
        )

    # conjunction sets and plain unordered sets
    merged_filters, _ = _merge_unordered(
        base.universe_filters, ours.universe_filters, theirs.universe_filters,
        key=lambda f: f.id,
    )
    merged_excluded, _ = _merge_unordered(
        base.excluded_assets, ours.excluded_assets, theirs.excluded_assets
    )

    structural = StructuralStatus.CONFLICTED if conflicts else StructuralStatus.CLEAN

    # Contract changes are not auto-merged: a contract is a promise to consumers,
    # so a divergent change requires a human decision.
    contract = ours.contract
    contract_status = ContractStatus.SATISFIED
    if ours.contract.to_json() != theirs.contract.to_json():
        if base.contract.to_json() == ours.contract.to_json():
            contract = theirs.contract
        elif base.contract.to_json() == theirs.contract.to_json():
            contract = ours.contract
        else:
            contract_status = ContractStatus.VIOLATED
            conflicts.append(
                Conflict(
                    field="contract",
                    semantics=Semantics.SCALAR,
                    detail="both sides changed the output contract; consumers are pinned to it",
                    ours=ours.contract.to_json(),
                    theirs=theirs.contract.to_json(),
                )
            )
            structural = StructuralStatus.CONFLICTED

    candidate = Methodology(
        concept=base.concept,
        version=max(ours.version, theirs.version) + 1,
        title=ours.title,
        objective=ours.objective,
        contract=contract,
        params=merged_params,
        rules=tuple(merged_rules.values()),
        pipeline=tuple(merged_pipeline or ()),
        universe_filters=tuple(
            sorted(merged_filters.values(), key=lambda f: f.id)
        ),
        excluded_assets=tuple(sorted(merged_excluded.values())),
        scoring_terms=merged_scores,
        fallback_chain=tuple(merged_fallback or ()),
        grounded_in=ours.grounded_in,
        assumptions=tuple(sorted(set(ours.assumptions) | set(theirs.assumptions))),
        limitations=tuple(sorted(set(ours.limitations) | set(theirs.limitations))),
        derived_from=f"{ours.version_id}+{theirs.version_id}",
        change_rationale=(
            f"merge of {ours.version_id} and {theirs.version_id}: "
            f"{ours.change_rationale} | {theirs.change_rationale}"
        ).strip(),
        risk_classification=ours.risk_classification,
        cost_model=ours.cost_model,
    )

    economic_status, problems, retests = _check_economics(candidate)
    notes.extend(problems)

    comparability, comp_notes = _check_comparability(base, candidate)
    notes.extend(comp_notes)
    if comparability is ComparabilityStatus.BROKEN:
        retests.append(
            "re-run all published results: prior figures describe a different strategy"
        )

    return MergeResult(
        structural_status=structural,
        contract_status=contract_status,
        economic_status=economic_status,
        comparability_status=comparability,
        merged=candidate if structural is StructuralStatus.CLEAN else None,
        unresolved_conflicts=conflicts,
        required_retests=sorted(set(retests)),
        notes=notes,
    )
