"""Quantify Methodology Specification 0.1 — the typed methodology AST.

A methodology is data, not a Python function. That distinction is the one
irreversible decision in the redesign: `merge()`, structural diff, version
identity and supersession are all downstream of it, and none is achievable if a
methodology is prose or an opaque callable.

Conventions follow `context-runtime`'s SPEC.md rather than inventing a second
set: RFC 2119 keywords, frozen dataclasses, a stable JSON form for anything
persisted, RFC 3339 UTC timestamps, and `<kind>_<ulid>`-shaped identifiers.

Two identifiers per family, following Zenodo/DataCite:

* **concept id** — ``methodology/hrp``, stable, always means "the method".
* **version id** — ``methodology/hrp@2``, immutable, means "these exact rules".

The canonical hash is computed over the semantic content only, so reformatting a
YAML file does not mint a new version but changing a threshold does.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

SPEC_VERSION = "0.1"


class Semantics(str, Enum):
    """How a node combines when two branches both modify it.

    This classification is the difference between a merge that is safe and one
    that is merely syntactically clean. Two categories (ordered vs unordered) are
    not enough — see `NORMALIZED_MAP` and `CONJUNCTION_SET`, both of which can
    merge without textual conflict and still produce an invalid strategy.
    """

    UNORDERED_SET = "unordered_set"
    """Bag semantics. Concurrent additions from both sides merge silently."""

    ORDERED_SEQUENCE = "ordered_sequence"
    """Order carries meaning. Concurrent edits to the same region conflict."""

    NORMALIZED_MAP = "normalized_map"
    """Weights. Merges cleanly but MUST be revalidated for normalization,
    leverage and concentration — two branches can each make a reasonable
    adjustment and jointly break the portfolio."""

    CONJUNCTION_SET = "conjunction_set"
    """Filters ANDed together. Merging is set union, but the union can select
    the empty universe, so the result MUST be checked for feasibility."""

    WEIGHTED_EXPRESSION = "weighted_expression"
    """Scoring terms whose relative magnitudes carry meaning. Rescaling one term
    changes the strategy even though no term was added or removed."""

    SCALAR = "scalar"
    """A single value. Concurrent differing edits conflict."""


#: Field name -> merge semantics. The merge layer refuses to operate on a field
#: absent from this table rather than guessing, so adding a node type to the
#: spec forces an explicit decision about how it combines.
FIELD_SEMANTICS: Mapping[str, Semantics] = {
    "rules": Semantics.UNORDERED_SET,
    "constraints": Semantics.UNORDERED_SET,
    "excluded_assets": Semantics.UNORDERED_SET,
    "pipeline": Semantics.ORDERED_SEQUENCE,
    "fallback_chain": Semantics.ORDERED_SEQUENCE,
    "weights": Semantics.NORMALIZED_MAP,
    "universe_filters": Semantics.CONJUNCTION_SET,
    "scoring_terms": Semantics.WEIGHTED_EXPRESSION,
    "params": Semantics.SCALAR,
}


class PerformanceClass(str, Enum):
    """What kind of number a result is. Required on every published figure.

    Rendering MUST refuse a performance record without one, and MUST refuse to
    combine records of different classes into a single series — GIPS forbids
    linking actual to theoretical performance, and a chart flowing seamlessly
    from backtest into live is the most damaging artifact in this product
    category.
    """

    BACKTEST_HYPOTHETICAL = "BACKTEST_HYPOTHETICAL"
    PAPER_LIVE_OOS = "PAPER_LIVE_OOS"
    ACTUAL_MANAGED = "ACTUAL_MANAGED"


#: Disclosure text bound to the data object, not the page template. A record
#: carrying BACKTEST_HYPOTHETICAL renders this wherever it is served — API
#: response, export, embed — so the number cannot be copied away from its caveat.
REQUIRED_DISCLOSURE: Mapping[PerformanceClass, str] = {
    PerformanceClass.BACKTEST_HYPOTHETICAL: (
        "Hypothetical backtested performance. No capital was managed and no orders "
        "were placed. Backtested results are hypothetical performance under SEC Rule "
        "206(4)-1(e), do not reflect actual trading, and are not a track record. "
        "Results depend on assumptions stated in the methodology, including "
        "transaction costs, and would differ under other assumptions."
    ),
    PerformanceClass.PAPER_LIVE_OOS: (
        "Paper-traded out-of-sample performance. Positions were recorded forward in "
        "time without capital at risk. No orders were placed. Not a track record; "
        "actual trading would incur costs and market impact not fully modelled here."
    ),
    PerformanceClass.ACTUAL_MANAGED: (
        "Actual performance of managed capital. Past performance does not predict "
        "future results."
    ),
}


@dataclass(frozen=True)
class Citation:
    """A source this methodology operationalizes."""

    identifier: str                       # doi:..., arxiv:..., ssrn:...
    title: str
    claim_used: str = ""                  # what was taken from it, specifically

    def to_json(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "claim_used": self.claim_used,
        }


@dataclass(frozen=True)
class Param:
    """A named, typed parameter. Values live here, never in module constants.

    The whole point of the AST is that "same method, different lookback" is a
    version bump rather than a source edit, which requires parameters to be data.
    """

    value: Any
    unit: str = ""
    description: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "description": self.description}


@dataclass(frozen=True)
class Rule:
    """A verifiable assertion about a declared field — not a computation.

    Rules used to carry a free-text ``expr`` that nothing evaluated. A reader saw
    ``max_asset_weight <= 0.25`` and reasonably concluded the constraint was
    enforced by that rule; it was not, and the two could silently disagree with
    the contract that actually enforced it.

    The resolution is that **contracts execute, rules verify.** A rule names the
    field that realizes it and the property that field must have. Nothing here
    performs work: the rule asserts that work declared elsewhere has the shape it
    claims, and the assertion is checkable.
    """

    id: str
    enforced_by: str
    """Dotted path to the realization, e.g. ``contract.weight_bounds.max``,
    ``params.max_turnover``, or ``pipeline.apply_turnover_cap``."""

    expected: str = "present"
    """Property the realization must have: ``<= 0.25``, ``>= 504``, ``== sample``
    or ``present``."""

    description: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enforced_by": self.enforced_by,
            "expected": self.expected,
            "description": self.description,
        }


@dataclass(frozen=True)
class UniverseFilter:
    """A declared restriction on the investable set, and what realizes it.

    Same defect as rules: ``has_full_lookback_history`` was a bare string that
    nothing applied. It is realized by ``estimate_correlation`` raising when the
    window is short — so the filter now names that stage rather than implying an
    unimplemented one.
    """

    id: str
    enforced_by: str
    description: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "enforced_by": self.enforced_by,
            "description": self.description,
        }


@dataclass(frozen=True)
class ConstraintPolicy:
    """Which constraints are inviolable and what happens when they conflict.

    Turnover caps and weight bounds can disagree: blending toward a previous
    holding to respect a turnover budget may carry a weight back over its
    ceiling. Resolving that silently in executor code is a hidden degree of
    freedom — two implementations could resolve it opposite ways and both claim
    to run the same methodology.

    Making precedence part of the executable contract means the resolution is
    declared, diffable, and auditable, and every override it licenses is counted.
    """

    hard: Sequence[str] = ("weight_bounds", "gross_leverage", "prohibited_assets")
    soft: Sequence[str] = ("turnover_cap",)
    soft_may_be_violated_to_satisfy_hard: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {
            "hard": list(self.hard),
            "soft": list(self.soft),
            "soft_may_be_violated_to_satisfy_hard": self.soft_may_be_violated_to_satisfy_hard,
        }


@dataclass(frozen=True)
class FallbackRule:
    """A declared degradation path.

    Previously a bare list of handler names resolved inside the executor, which
    meant a run could spend a large share of its evaluation period allocating by
    a rule the methodology never described, while still being reported as that
    methodology. Declaring the trigger, the rule, its constraints and the
    disclosure makes the degradation part of the published artifact — and lets a
    run report how often it was used.
    """

    trigger: str                       # e.g. "insufficient_history"
    allocation_rule: str               # e.g. "inverse_volatility"
    constraints: Sequence[str] = ()    # which contract constraints still apply
    disclosure: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger,
            "allocation_rule": self.allocation_rule,
            "constraints": list(self.constraints),
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class ContractBreak:
    """One structured reason two contracts are not interchangeable.

    The engine has always known *which field* moved and *why that matters*; until
    now it flattened both into one sentence at the point of detection. Keeping the
    parts separate lets an interface show the difference and its consequence in
    different places without re-parsing prose — and stops each page inventing its
    own explanation of what a rebalance-frequency change means.
    """

    field: str
    before: str
    after: str
    why: str

    def describe(self) -> str:
        """The legacy one-line form. Callers reading strings still work."""
        if self.before == "" or self.after == "":
            return f"{self.field}{':' if self.after else ''} {self.after or self.before}".strip()
        return f"{self.field} {self.before} -> {self.after}"

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "before": self.before,
                "after": self.after, "why": self.why, "describe": self.describe()}


@dataclass(frozen=True)
class OutputContract:
    """The interface a methodology promises to its consumers.

    Enforced, in dbt's sense: a change that breaks this contract is not a patch,
    it MUST mint a new version. Without that, a downstream mission can silently
    receive incomparable results from the same concept id.
    """

    universe: Sequence[str]
    rebalance_frequency: str              # e.g. "5B" — 5 business days
    weight_bounds: Dict[str, float] = field(default_factory=lambda: {"min": 0.0, "max": 1.0})
    gross_leverage_max: float = 1.0
    requires_cost_model: bool = True
    constraint_policy: ConstraintPolicy = field(default_factory=ConstraintPolicy)

    def to_json(self) -> Dict[str, Any]:
        return {
            "universe": list(self.universe),
            "rebalance_frequency": self.rebalance_frequency,
            "weight_bounds": dict(self.weight_bounds),
            "gross_leverage_max": self.gross_leverage_max,
            "requires_cost_model": self.requires_cost_model,
            "constraint_policy": self.constraint_policy.to_json(),
        }

    def compatibility_breaks(self, other: "OutputContract") -> List[ContractBreak]:
        """Structured reasons `self` is not a drop-in replacement for `other`.

        A non-empty result means consumers pinned to `other` cannot be moved
        without review — which is exactly the condition that forces a new version
        rather than an in-place edit.

        Each break carries the consequence alongside the change, because a field
        name alone does not tell a reader why two published figures cannot sit in
        the same table.
        """
        breaks: List[ContractBreak] = []
        if set(self.universe) != set(other.universe):
            added = sorted(set(self.universe) - set(other.universe))
            removed = sorted(set(other.universe) - set(self.universe))
            if removed:
                breaks.append(ContractBreak(
                    "universe removed", "", str(removed),
                    "The later version can no longer hold instruments the earlier "
                    "one could, so their returns come from different opportunity sets.",
                ))
            if added:
                breaks.append(ContractBreak(
                    "universe added", "", str(added),
                    "The later version can hold instruments the earlier one never "
                    "saw; any improvement may be the new instruments rather than "
                    "the method.",
                ))
        if self.rebalance_frequency != other.rebalance_frequency:
            breaks.append(ContractBreak(
                "rebalance frequency", other.rebalance_frequency,
                self.rebalance_frequency,
                "Turnover and cost drag scale with rebalance frequency, so the two "
                "sets of figures were produced under different cost burdens.",
            ))
        if self.gross_leverage_max != other.gross_leverage_max:
            breaks.append(ContractBreak(
                "gross leverage", str(other.gross_leverage_max),
                str(self.gross_leverage_max),
                "A different leverage ceiling changes the achievable risk level, so "
                "returns are not on the same scale.",
            ))
        # Weight bounds constrain what the strategy can hold at all, so a change
        # here means prior results describe a strategy that could take positions
        # this one cannot — or vice versa.
        for edge in ("min", "max"):
            mine = float(self.weight_bounds.get(edge, 0.0 if edge == "min" else 1.0))
            theirs = float(other.weight_bounds.get(edge, 0.0 if edge == "min" else 1.0))
            if mine != theirs:
                breaks.append(ContractBreak(
                    f"weight bound {edge}", str(theirs), str(mine),
                    "The earlier version could take positions this one cannot (or "
                    "vice versa), so the two describe different strategies rather "
                    "than the same strategy tuned differently.",
                ))
        if other.requires_cost_model and not self.requires_cost_model:
            breaks.append(ContractBreak(
                "cost model requirement dropped", "", "",
                "One set of figures is net of trading costs and the other is not; "
                "the gap between them is not skill.",
            ))
        return breaks

    def breaks_compatibility_with(self, other: "OutputContract") -> List[str]:
        """One-line forms of `compatibility_breaks`, for callers reading prose."""
        return [b.describe() for b in self.compatibility_breaks(other)]


@dataclass(frozen=True)
class Methodology:
    """A versioned, executable investment methodology.

    `concept` is stable across versions; `version` increments. Together they form
    the version id. `content_hash` is derived, never stored by hand.
    """

    concept: str
    version: int
    title: str
    objective: str
    contract: OutputContract
    params: Dict[str, Param] = field(default_factory=dict)
    rules: Sequence[Rule] = ()
    pipeline: Sequence[str] = ()
    universe_filters: Sequence[UniverseFilter] = ()
    excluded_assets: Sequence[str] = ()
    scoring_terms: Dict[str, float] = field(default_factory=dict)
    fallback_chain: Sequence[str] = ()
    fallbacks: Sequence[FallbackRule] = ()
    grounded_in: Sequence[Citation] = ()
    claims_ref: Sequence[str] = ()
    """References to `claim/<name>@<version>` artifacts this methodology rests on.

    References rather than prose, so a claim can be supported by two
    methodologies, contradicted by a replication, or superseded — without editing
    every file that quotes it."""

    assumptions_ref: Sequence[str] = ()
    """References to `assumption/<name>@<version>` artifacts. Distinct from the
    prose `assumptions` list, which stays for narrative context."""
    assumptions: Sequence[str] = ()
    limitations: Sequence[str] = ()
    derived_from: Optional[str] = None    # version id of the parent
    change_rationale: str = ""
    risk_classification: str = "medium"   # SR 26-2 inventory field
    cost_model: str = "flat_bps"
    deprecation_date: Optional[str] = None
    spec_version: str = SPEC_VERSION

    # ---- identity ---------------------------------------------------------

    @property
    def concept_id(self) -> str:
        """Stable identifier for the method across all versions."""
        return f"methodology/{self.concept}"

    @property
    def version_id(self) -> str:
        """Immutable identifier for these exact rules."""
        return f"methodology/{self.concept}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        """The semantic content, in a deterministic shape.

        Excludes documentation-only fields (`title`, `change_rationale`,
        `deprecation_date`) so that editing prose does not mint a new version,
        and includes everything that changes behaviour so that a threshold edit
        does. Keys are sorted at serialization time.
        """
        return {
            "spec_version": self.spec_version,
            "concept": self.concept,
            "version": self.version,
            "objective": self.objective,
            "contract": self.contract.to_json(),
            "params": {k: self.params[k].to_json() for k in sorted(self.params)},
            "rules": sorted((r.to_json() for r in self.rules), key=lambda r: r["id"]),
            "pipeline": list(self.pipeline),
            "universe_filters": sorted(
                (f.to_json() for f in self.universe_filters), key=lambda f: f["id"]
            ),
            "excluded_assets": sorted(self.excluded_assets),
            "scoring_terms": {k: self.scoring_terms[k] for k in sorted(self.scoring_terms)},
            "fallback_chain": list(self.fallback_chain),
            "fallbacks": [f.to_json() for f in self.fallbacks],
            "grounded_in": sorted(
                (c.to_json() for c in self.grounded_in), key=lambda c: c["identifier"]
            ),
            "claims_ref": sorted(self.claims_ref),
            "assumptions_ref": sorted(self.assumptions_ref),
            "assumptions": sorted(self.assumptions),
            "limitations": sorted(self.limitations),
            "cost_model": self.cost_model,
            "risk_classification": self.risk_classification,
        }

    def canonical_json(self) -> str:
        """Deterministic serialization. This is what gets hashed."""
        return json.dumps(
            self.canonical_form(), sort_keys=True, separators=(",", ":"), default=str
        )

    @property
    def content_hash(self) -> str:
        """sha256 of the canonical form. Two methodologies with the same hash
        are the same methodology regardless of file formatting."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    # ---- revision ---------------------------------------------------------

    def revise(self, *, change_rationale: str, **changes: Any) -> "Methodology":
        """Produce the next version, recording lineage.

        Refuses an empty rationale: a version whose reason for existing is not
        recorded cannot be reviewed later, and the changelog is the artifact that
        makes supersession legible.
        """
        if not change_rationale.strip():
            raise ValueError("change_rationale is required — an unexplained version cannot be reviewed")
        return replace(
            self,
            version=self.version + 1,
            derived_from=self.version_id,
            change_rationale=change_rationale,
            **changes,
        )

    def to_json(self) -> Dict[str, Any]:
        """Full serialization, including documentation fields and derived ids."""
        payload = self.canonical_form()
        payload.update(
            {
                "concept_id": self.concept_id,
                "version_id": self.version_id,
                "content_hash": self.content_hash,
                "title": self.title,
                "derived_from": self.derived_from,
                "change_rationale": self.change_rationale,
                "deprecation_date": self.deprecation_date,
            }
        )
        return payload


def from_dict(payload: Mapping[str, Any]) -> Methodology:
    """Parse a methodology from its JSON/YAML form.

    Unknown top-level keys are preserved by being ignored rather than rejected,
    matching the SPEC's forward-compatibility rule: a v0.1 reader MUST round-trip
    documents carrying v0.2+ fields it does not understand.
    """
    contract_raw = payload.get("contract", {})
    policy_raw = contract_raw.get("constraint_policy") or {}
    contract = OutputContract(
        universe=tuple(contract_raw.get("universe", ())),
        rebalance_frequency=contract_raw.get("rebalance_frequency", "5B"),
        weight_bounds=dict(contract_raw.get("weight_bounds", {"min": 0.0, "max": 1.0})),
        gross_leverage_max=float(contract_raw.get("gross_leverage_max", 1.0)),
        requires_cost_model=bool(contract_raw.get("requires_cost_model", True)),
        constraint_policy=ConstraintPolicy(
            hard=tuple(policy_raw.get("hard", ConstraintPolicy().hard)),
            soft=tuple(policy_raw.get("soft", ConstraintPolicy().soft)),
            soft_may_be_violated_to_satisfy_hard=bool(
                policy_raw.get("soft_may_be_violated_to_satisfy_hard", True)
            ),
        ),
    )
    return Methodology(
        concept=payload["concept"],
        version=int(payload["version"]),
        title=payload.get("title", payload["concept"]),
        objective=payload.get("objective", ""),
        contract=contract,
        params={
            k: Param(**v) if isinstance(v, dict) else Param(value=v)
            for k, v in (payload.get("params") or {}).items()
        },
        rules=tuple(Rule(**r) for r in (payload.get("rules") or ())),
        pipeline=tuple(payload.get("pipeline") or ()),
        universe_filters=tuple(
            UniverseFilter(**f) for f in (payload.get("universe_filters") or ())
        ),
        excluded_assets=tuple(payload.get("excluded_assets") or ()),
        scoring_terms=dict(payload.get("scoring_terms") or {}),
        fallback_chain=tuple(payload.get("fallback_chain") or ()),
        fallbacks=tuple(FallbackRule(**f) for f in (payload.get("fallbacks") or ())),
        grounded_in=tuple(Citation(**c) for c in (payload.get("grounded_in") or ())),
        claims_ref=tuple(payload.get("claims_ref") or ()),
        assumptions_ref=tuple(payload.get("assumptions_ref") or ()),
        assumptions=tuple(payload.get("assumptions") or ()),
        limitations=tuple(payload.get("limitations") or ()),
        derived_from=payload.get("derived_from"),
        change_rationale=payload.get("change_rationale", ""),
        risk_classification=payload.get("risk_classification", "medium"),
        cost_model=payload.get("cost_model", "flat_bps"),
        deprecation_date=payload.get("deprecation_date"),
        spec_version=payload.get("spec_version", SPEC_VERSION),
    )
