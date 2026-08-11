"""What may differ between two results, declared rather than curated.

`ISOLATION_DIMENSIONS` was the last hand-maintained causal table in the system: a
tuple of strings that had to be remembered whenever a new runtime or scenario
field appeared. It was wrong twice already — `allocation_rule` and
`data_snapshot` were both missing, and both silently let a comparison claim
attribution it did not have.

The fix is not to derive it from runtime kinds. Causal dimensions come from two
different classes of artifact and forcing scenario fields to masquerade as
runtimes would be the wrong abstraction:

    runtime           reusable, public, versioned policy  (tax, calendar, data)
    scenario artifact one person's declaration            (flow schedule, capital)

So both register into one dimension registry, each stating which comparison
classes it participates in. A new dimension that declares nothing fails the
suite rather than defaulting into or out of causal comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


class SourceKind(str, Enum):
    RUNTIME = "runtime"
    """A versioned policy artifact. Reusable and impersonal."""

    SCENARIO_ARTIFACT = "scenario_artifact"
    """One person's declaration. Never reusable, never public."""


class Requirement(str, Enum):
    """What a comparison class demands of a dimension."""

    MUST_EQUAL = "must_equal"
    """Differing here defeats this class's attribution claim."""

    MAY_DIFFER = "may_differ"
    """Differing here is allowed and is part of what the class describes."""

    DEFEATS_COMPARISON = "defeats_comparison"
    """Differing here means the two have no shared basis at all — not a weaker
    comparison, no comparison. Distinct from MUST_EQUAL, which downgrades the
    class rather than voiding it."""


@dataclass(frozen=True)
class ComparisonDimension:
    """One axis along which two results can differ."""

    id: str
    source_kind: SourceKind
    causal_label: str
    """Plain language, for the sentence a reader gets. "cash-flow schedule",
    not "flow_schedule"."""

    supports: Mapping[str, Requirement]
    extractor: Optional[Callable[[Any, Any], Any]] = None
    """`(scenario, environment) -> comparable value`. Optional so a dimension can
    be declared before its plumbing exists — but a declared dimension with no
    extractor is reported, not silently skipped."""

    required: bool = True
    isolation_eligible: bool = True
    """Whether this dimension can ever be the isolated cause. A title correction
    is a difference and is never an explanation."""

    runtime_kind: Optional[str] = None
    """The runtime that gives this dimension its meaning.

    Distinct from `source_kind`, which says where the *value* comes from. A
    flow schedule is a personal declaration (`SCENARIO_ARTIFACT`) whose
    semantics come from the flow runtime, so it is both. Set this and
    `depends_on` is derived from that runtime rather than written twice."""

    depends_on: Sequence[str] = ()
    """Dimensions whose equality this one's meaning requires.

    Tax means different things under different accounts, so "only tax differs"
    is a claim about causation *only if* account is also equal.

    **Derived where possible.** A dimension naming a `runtime_kind` inherits
    this from that runtime's `causal_dependencies()`, so the fact lives in one
    place. `reconcile_dependencies()` checks the two agree, and the check fails
    if a runtime later adds a semantic precondition the registry never heard
    about."""

    disclosure_template: str = ""

    def requirement_for(self, comparison_class: str) -> Requirement:
        return self.supports.get(comparison_class, Requirement.MUST_EQUAL)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id, "source_kind": self.source_kind.value,
            "causal_label": self.causal_label,
            "supports": {k: v.value for k, v in self.supports.items()},
            "required": self.required,
            "isolation_eligible": self.isolation_eligible,
            "depends_on": list(self.depends_on),
            "has_extractor": self.extractor is not None,
            "disclosure_template": self.disclosure_template,
        }


REGISTRY: Dict[str, ComparisonDimension] = {}


class UndeclaredDimension(KeyError):
    """A field participates in comparison without saying how."""


def register(dimension: ComparisonDimension) -> ComparisonDimension:
    if dimension.id in REGISTRY:
        raise ValueError(f"dimension {dimension.id!r} is already registered")
    REGISTRY[dimension.id] = dimension
    return dimension


def dimension(dimension_id: str) -> ComparisonDimension:
    if dimension_id not in REGISTRY:
        raise UndeclaredDimension(
            f"{dimension_id!r} is not a registered comparison dimension. A field "
            "that can differ between two results must declare which comparison "
            "classes it participates in — defaulting either way is how a "
            "comparison comes to claim attribution it does not have"
        )
    return REGISTRY[dimension_id]


def dimensions_requiring_equality(comparison_class: str) -> List[str]:
    """Everything that must match for a class's attribution claim to hold."""
    return sorted(
        d.id for d in REGISTRY.values()
        if d.requirement_for(comparison_class) is Requirement.MUST_EQUAL
    )


def dimensions_defeating(comparison_class: str) -> List[str]:
    return sorted(
        d.id for d in REGISTRY.values()
        if d.requirement_for(comparison_class) is Requirement.DEFEATS_COMPARISON
    )


def isolation_of(differing: Sequence[str]) -> Optional[str]:
    """The single dimension that explains a difference, if there is one.

    Two conditions, and the second is the one a naive implementation misses:

    1. exactly one dimension differs, and it is eligible to be a cause;
    2. every dimension that one *depends on* is equal.

    Without (2), "only tax differs" can be reported as a cause when tax's
    meaning was jointly determined by an account that also differs — an
    explanation assembled from a dimension whose semantics were never checked.
    """
    candidates = [d for d in differing]
    if len(candidates) != 1:
        return None

    only = candidates[0]
    if only not in REGISTRY or not REGISTRY[only].isolation_eligible:
        return None
    if any(dep in differing for dep in REGISTRY[only].depends_on):
        return None
    return only


#: Runtime kind -> the dimension that represents it. Built from the registry so
#: a runtime's declarations can be translated into dimension ids.
def _dimension_for_runtime(kind: str) -> Optional[str]:
    for spec in REGISTRY.values():
        if spec.runtime_kind == kind and spec.source_kind is SourceKind.RUNTIME:
            return spec.id
    return None


def derived_dependencies(spec: ComparisonDimension) -> Optional[List[str]]:
    """What this dimension's runtime says must be equal, as dimension ids.

    Returns None when the dimension names no runtime, in which case its
    `depends_on` is authored rather than derived and cannot be cross-checked.
    """
    if spec.runtime_kind is None:
        return None
    try:
        from ..runtime import RUNTIME_TYPES
    except ImportError:  # pragma: no cover - runtime package optional
        return None

    runtime_type = RUNTIME_TYPES.get(spec.runtime_kind)
    if runtime_type is None:
        return None  # reported by `unreconcilable`, not silently accepted

    out = []
    for required_kind in runtime_type.causal_dependencies():
        mapped = _dimension_for_runtime(required_kind)
        if mapped:
            out.append(mapped)
    return sorted(out)


def reconcile_dependencies() -> Dict[str, Dict[str, List[str]]]:
    """Where the registry and the runtimes disagree about causal dependence.

    Two declarations of one fact drift. This is the same reason
    `ISOLATION_DIMENSIONS` stopped being a hand-written tuple — except here the
    drift would be silent, because a comparison would simply stop checking
    something and still report `attribution_isolated`.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for spec in REGISTRY.values():
        derived = derived_dependencies(spec)
        if derived is None:
            continue
        declared = sorted(spec.depends_on)
        if declared != derived:
            out[spec.id] = {"declared": declared, "derived": derived}
    return out


def unreconcilable() -> Dict[str, str]:
    """Dimensions naming a runtime kind that has no registered runtime type.

    The derivation quietly does nothing for these, which is worse than not
    attempting it: the registry looks reconciled while one dimension's
    dependencies are still hand-authored and unchecked. Reported so the gap is
    visible rather than inferred from an empty disagreement list.
    """
    from ..runtime import RUNTIME_TYPES

    return {
        spec.id: spec.runtime_kind
        for spec in REGISTRY.values()
        if spec.runtime_kind is not None and spec.runtime_kind not in RUNTIME_TYPES
    }


def unextractable() -> List[str]:
    """Dimensions declared without a way to read them.

    A declaration with no realization, in the one place it would be least
    visible: the comparison would simply never notice the dimension differed.
    """
    return sorted(d.id for d in REGISTRY.values() if d.extractor is None)


# --- the registry -----------------------------------------------------------
#
# STRATEGY_EFFECT holds everything constant so a difference is the rule.
# PERSONAL_OUTCOME lets the person's own declarations vary — that is what it is
# for — while still refusing a comparison across different periods or data.
# CONSTRAINT_EFFECT lets only execution timing move.

_STRATEGY, _PERSONAL, _CONSTRAINT = (
    "STRATEGY_EFFECT", "PERSONAL_OUTCOME", "CONSTRAINT_EFFECT")


def _runtime_hash(kind: str):
    def extract(scenario: Any, environment: Any) -> Any:
        runtime = getattr(environment, "runtimes", {}).get(kind)
        return runtime.compatibility_hash if runtime is not None else None
    return extract


register(ComparisonDimension(
    id="allocation_rule", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="what the money buys",
    extractor=lambda scenario, env: getattr(scenario, "rule_hash", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
    disclosure_template="The two bought different things.",
))

register(ComparisonDimension(
    id="flow_schedule", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="cash-flow schedule",
    runtime_kind="flow",
    extractor=lambda scenario, env: getattr(
        getattr(scenario, "flow_schedule", None), "schedule_hash", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
    depends_on=("calendar",),
    disclosure_template="Money arrived on a different schedule.",
))

register(ComparisonDimension(
    id="starting_capital", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="starting capital",
    extractor=lambda scenario, env: getattr(
        getattr(scenario, "flow_schedule", None), "starting_capital", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

register(ComparisonDimension(
    id="cash_policy", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="what uninvested cash earns",
    extractor=lambda scenario, env: getattr(scenario, "cash_policy_ref", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

register(ComparisonDimension(
    id="tax_treatment", source_kind=SourceKind.RUNTIME,
    runtime_kind="tax",
    causal_label="tax treatment",
    extractor=_runtime_hash("tax"),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
    depends_on=("account",),
    disclosure_template="Gains and withholding were treated differently.",
))

register(ComparisonDimension(
    id="account", source_kind=SourceKind.RUNTIME,
    runtime_kind="account",
    causal_label="account rules",
    extractor=_runtime_hash("account"),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

register(ComparisonDimension(
    id="fees", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="trading costs",
    extractor=lambda scenario, env: getattr(scenario, "cost_bps", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

register(ComparisonDimension(
    id="execution_timing", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="when orders executed",
    extractor=lambda scenario, env: getattr(scenario, "execution_lag", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MAY_DIFFER},
    disclosure_template="Orders executed at different times.",
))

register(ComparisonDimension(
    id="evaluation_period", source_kind=SourceKind.SCENARIO_ARTIFACT,
    causal_label="the period measured",
    extractor=lambda scenario, env: (
        getattr(scenario, "period_start", None), getattr(scenario, "period_end", None)),
    supports={_STRATEGY: Requirement.DEFEATS_COMPARISON,
              _PERSONAL: Requirement.DEFEATS_COMPARISON,
              _CONSTRAINT: Requirement.DEFEATS_COMPARISON},
    isolation_eligible=False,
    disclosure_template=("They were measured over different periods, so they "
                         "were exposed to different markets rather than to "
                         "different rules."),
))

register(ComparisonDimension(
    id="data_snapshot", source_kind=SourceKind.RUNTIME,
    causal_label="the data that was served",
    extractor=lambda scenario, env: getattr(scenario, "data_snapshot", None),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
    depends_on=("market_data",),
    disclosure_template=("A vendor revision changed the answer without anyone "
                         "changing a decision."),
))

register(ComparisonDimension(
    id="market_data", source_kind=SourceKind.RUNTIME,
    runtime_kind="market_data",
    causal_label="how data is sourced and read",
    extractor=_runtime_hash("market_data"),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

register(ComparisonDimension(
    id="calendar", source_kind=SourceKind.RUNTIME,
    runtime_kind="calendar",
    causal_label="the trading calendar",
    extractor=_runtime_hash("calendar"),
    supports={_STRATEGY: Requirement.MUST_EQUAL,
              _PERSONAL: Requirement.MAY_DIFFER,
              _CONSTRAINT: Requirement.MUST_EQUAL},
))

#: Derived, not curated. The tuple `ISOLATION_DIMENSIONS` used to be.
ISOLATION_DIMENSIONS = tuple(dimensions_requiring_equality(_CONSTRAINT)) + tuple(
    dimensions_defeating(_CONSTRAINT))
