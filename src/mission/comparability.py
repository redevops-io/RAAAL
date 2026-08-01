"""Two kinds of comparison, and the one that cannot attribute a winner.

`cash_flow_schedule_equivalence` alone is too blunt. It would refuse the
comparison a user most wants — *"was I better off contributing monthly or
investing my year-end bonus?"* — which is a legitimate question with a real
answer. The problem is not that the comparison is invalid; it is that its
difference is **not attributable to the investment rule**.

So a verdict carries a class:

    STRATEGY_EFFECT     schedules identical; the rule is isolated
    PERSONAL_OUTCOME    schedules differ; the difference is rule + timing + size

Both are comparable. Only the first supports a sentence containing the word
"better strategy", and `attribution_isolated` is the field that says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..comparison.dimensions import REGISTRY as REGISTERED_DIMENSIONS
from ..comparison.dimensions import dimension as comparison_dimension
from ..comparison.dimensions import isolation_of


class ComparisonClass(str, Enum):
    STRATEGY_EFFECT = "STRATEGY_EFFECT"
    """Everything except the rule is held identical, so a difference is the rule."""

    PERSONAL_OUTCOME = "PERSONAL_OUTCOME"
    """Schedules or conditions differ. A real comparison of two lived outcomes,
    and not a statement about which rule is better."""

    CONSTRAINT_EFFECT = "CONSTRAINT_EFFECT"
    """Same plan, same schedule, and the only difference is a constraint the
    person did not choose — a blackout window, a data gap, a delayed fill.

    The difference is attributable, but to the constraint rather than to the
    rule. Without this class a counterfactual would be reported as merely
    'personal outcome', which understates it: what a blackout cost is a specific
    number and the person could not have avoided it."""


#: Derived from the comparison-dimension registry, not maintained here.
#:
#: This was a hand-curated tuple, and it was wrong twice — `allocation_rule` and
#: `data_snapshot` were both missing, and both silently let a comparison claim
#: attribution it did not have. A list that must be remembered is a list that
#: will be forgotten, so the dimensions now declare their own participation and
#: this is read from them.
ISOLATION_DIMENSIONS = tuple(sorted(REGISTERED_DIMENSIONS))


#: Disclosure text bound to the verdict, not to a page. A sentence that lives in
#: page copy is absent from the API response, the CSV export and the saved
#: result — which are exactly the places a figure gets quoted without its caveat.
#: Versioned, because changing what a comparison claims is a change to the claim.
DISCLOSURE_VERSION = "comparability-disclosure@1"

#: How a verdict decided what "matched" means. Persisted with every verdict,
#: because the answer changed and old verdicts must keep meaning what they meant.
#:
#:     @1  two absent values compare equal, and the dimension is *also* listed
#:         as unchecked. Honest in the record, misleading when read as a match.
#:     @2  an absent value on either side is NOT_EVALUATED. A verdict claims
#:         "checked and equivalent", and when both hashes are empty that claim
#:         is false.
#:
#: `@2` additionally refuses `attribution_isolated` while any dimension required
#: to be equal was never evaluated: a comparison may still be shown, but it
#: cannot claim the strategy was isolated.
CLASSIFIER_VERSION = "comparability/classifier@2"


class DimensionStatus(str, Enum):
    MATCHED = "MATCHED"
    NOT_MATCHED = "NOT_MATCHED"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class DimensionResult:
    """One dimension, with enough to explain the verdict rather than assert it."""

    dimension: str
    status: DimensionStatus
    left_value: Any = None
    right_value: Any = None
    reason: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "status": self.status.value,
                "left_value": _readable(self.left_value),
                "right_value": _readable(self.right_value),
                "reason": self.reason}


def _readable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _absent(value: Any) -> bool:
    return value is None or value == ""

DISCLOSURES = {
    "STRATEGY_EFFECT": (
        "Every dimension outside the investment rule was held identical — "
        "contributions, starting capital, cash treatment, tax treatment, fees, "
        "execution timing and evaluation period. A difference between these "
        "figures is attributable to the rule."
    ),
    "PERSONAL_OUTCOME": (
        "The difference combines contribution timing, contribution size, and "
        "strategy behaviour. It does not isolate which strategy is better."
    ),
    "NOT_COMPARABLE": (
        "These were measured under conditions that share no common basis, so "
        "the figures cannot be read against each other."
    ),
    "CONSTRAINT_EFFECT": (
        "The plan, the schedule and the rule are identical. The difference is "
        "what the named constraint cost, not evidence about the strategy."
    ),
}


@dataclass(frozen=True)
class ComparabilityVerdict:
    """Whether two runs can be compared, and what a difference would mean."""

    comparison_class: ComparisonClass
    comparable: bool
    attribution_isolated: bool
    differing_dimensions: Sequence[str]
    detail: str = ""

    isolates: str = ""
    classifier_version: str = CLASSIFIER_VERSION
    dimension_results: Sequence["DimensionResult"] = ()
    """Per-dimension detail. A boolean said only whether *something* differed,
    which cannot distinguish "checked and equal" from "never looked at"."""

    unchecked_dimensions: Sequence[str] = ()
    """Registered dimensions these conditions could not report on.

    An unchecked dimension is a hole in the attribution claim, and the only
    thing worse than a hole is one nobody can see. Reported rather than assumed
    equal.
    """
    """What the difference *is* attributable to, when it is attributable to one
    thing. A boolean said only whether attribution was possible, never to what,
    which stopped being enough once a constraint could be the answer."""

    @property
    def disclosure_key(self) -> str:
        if not self.comparable:
            return "NOT_COMPARABLE"
        if self.comparison_class is ComparisonClass.CONSTRAINT_EFFECT:
            return "CONSTRAINT_EFFECT"
        return ("STRATEGY_EFFECT" if self.attribution_isolated
                else "PERSONAL_OUTCOME")

    @property
    def required_disclosure(self) -> str:
        """Travels with the verdict into API, UI, export and saved result."""
        return DISCLOSURES[self.disclosure_key]

    def what_a_difference_means(self) -> str:
        base = self.required_disclosure
        if self.comparable and not self.attribution_isolated:
            return f"These differ in {', '.join(self.differing_dimensions)}. {base}"
        return base

    def to_json(self) -> Dict[str, Any]:
        return {
            "class": self.comparison_class.value,
            "comparable": self.comparable,
            "attribution_isolated": self.attribution_isolated,
            "differing_dimensions": list(self.differing_dimensions),
            "differing_labels": [
                comparison_dimension(d).causal_label
                for d in self.differing_dimensions if d in REGISTERED_DIMENSIONS
            ],
            "isolates": self.isolates,
            "unchecked_dimensions": list(self.unchecked_dimensions),
            "classifier_version": self.classifier_version,
            "dimensions": [d.to_json() for d in self.dimension_results],
            "detail": self.detail,
            "required_disclosure": self.required_disclosure,
            "disclosure_version": DISCLOSURE_VERSION,
            "what_a_difference_means": self.what_a_difference_means(),
        }


@dataclass(frozen=True)
class RunConditions:
    """The dimensions a comparison must hold constant to isolate the rule."""

    flow_schedule_hash: str
    starting_capital: float
    cash_policy_rate: float
    tax_treatment: str
    cost_bps: float
    execution_lag: int
    period_start: str
    period_end: str
    allocation_rule_hash: str = ""
    account_hash: str = ""
    """The account runtime's compatibility hash. Empty on both sides means the
    dimension was not exercised, which compares equal — an honest "not checked
    here" rather than a silent omission from the registry."""

    calendar_hash: str = ""
    market_data_hash: str = ""
    """What the money buys. Two runs differing here differ in the rule itself,
    which defeats a constraint counterfactual even when everything else matches."""

    data_snapshot: str = ""
    """Which vintage of prices produced this. A restated series changes the
    answer without changing anything the user did, and a comparison across two
    snapshots silently attributes a data revision to a decision."""

    def dimension_map(self) -> Dict[str, Any]:
        return {
            "allocation_rule": self.allocation_rule_hash,
            "account": self.account_hash,
            "calendar": self.calendar_hash,
            "market_data": self.market_data_hash,
            "data_snapshot": self.data_snapshot,
            "flow_schedule": self.flow_schedule_hash,
            "starting_capital": self.starting_capital,
            "cash_policy": self.cash_policy_rate,
            "tax_treatment": self.tax_treatment,
            "fees": self.cost_bps,
            "execution_timing": self.execution_lag,
            "evaluation_period": (self.period_start, self.period_end),
        }


def classify_counterfactual(actual: RunConditions, counterfactual: RunConditions,
                            *, constraint: str) -> ComparabilityVerdict:
    """What a constraint cost, holding the plan and schedule fixed.

    "Three proposals expired while the trading window was shut — what would have
    happened had they executed on the first eligible day?" is a legitimate and
    useful question, and it is not a question about strategy. Routing it through
    `classify` would return PERSONAL_OUTCOME and attach the disclosure saying the
    difference does not identify a better strategy — true, and beside the point.
    """
    a, b = actual.dimension_map(), counterfactual.dimension_map()
    checkable = [d for d in ISOLATION_DIMENSIONS if d in a and d in b]
    differing = [d for d in checkable if a[d] != b[d]]

    if not differing:
        return ComparabilityVerdict(
            comparison_class=ComparisonClass.CONSTRAINT_EFFECT,
            comparable=True, attribution_isolated=True, differing_dimensions=(),
            detail=f"identical conditions; {constraint} did not bind",
            isolates=constraint,
        )
    if isolation_of(differing) == "execution_timing":
        return ComparabilityVerdict(
            comparison_class=ComparisonClass.CONSTRAINT_EFFECT,
            comparable=True, attribution_isolated=True,
            differing_dimensions=("execution_timing",),
            detail=f"only the timing forced by {constraint} differs",
            isolates=constraint,
        )

    # Anything else and the counterfactual has changed more than the constraint,
    # so it is no longer measuring the constraint.
    verdict = classify(actual, counterfactual)
    return ComparabilityVerdict(
        comparison_class=verdict.comparison_class,
        comparable=verdict.comparable,
        attribution_isolated=False,
        differing_dimensions=verdict.differing_dimensions,
        detail=(f"this counterfactual changes {', '.join(differing)} as well as "
                f"the timing, so it no longer measures what {constraint} cost"),
        isolates="",
    )


def classify(left: RunConditions, right: RunConditions) -> ComparabilityVerdict:
    """Compare two runs' conditions and say what a difference between them means.

    Note what is *not* here: neither result. A comparability verdict computed from
    outcomes would be a verdict about which answer is convenient.
    """
    a, b = left.dimension_map(), right.dimension_map()

    # Classifier @2. An absent value on either side is NOT_EVALUATED rather than
    # a match: a stored verdict claims "these were checked and found
    # equivalent", and when both hashes are empty that claim is false. @1
    # compared two absences as equal while also listing the dimension as
    # unchecked — honest in the record and misleading when read as a match.
    results: List[DimensionResult] = []
    for dimension in ISOLATION_DIMENSIONS:
        left_value, right_value = a.get(dimension), b.get(dimension)
        if _absent(left_value) or _absent(right_value):
            results.append(DimensionResult(
                dimension, DimensionStatus.NOT_EVALUATED, left_value, right_value,
                reason=("no value was pinned on "
                        + ("both sides" if _absent(left_value) and _absent(right_value)
                           else "one side")
                        + ", so nothing was compared")))
        elif left_value != right_value:
            results.append(DimensionResult(
                dimension, DimensionStatus.NOT_MATCHED, left_value, right_value,
                reason="the two runs pinned different values"))
        else:
            results.append(DimensionResult(
                dimension, DimensionStatus.MATCHED, left_value, right_value,
                reason="both runs pinned the same value"))

    by_status = {r.dimension: r.status for r in results}
    checkable = [d for d, s in by_status.items()
                 if s is not DimensionStatus.NOT_EVALUATED]
    unchecked = tuple(sorted(d for d, s in by_status.items()
                             if s is DimensionStatus.NOT_EVALUATED))
    differing = [d for d in checkable
                 if by_status[d] is DimensionStatus.NOT_MATCHED]

    if not differing:
        # Isolation requires that every dimension outside the rule was actually
        # checked. An unevaluated one is a hole in the attribution claim, so the
        # comparison is still shown and the claim is not made.
        isolated = not unchecked
        return ComparabilityVerdict(
            comparison_class=ComparisonClass.STRATEGY_EFFECT,
            comparable=True, attribution_isolated=isolated,
            differing_dimensions=(),
            detail=("every dimension outside the rule is identical" if isolated
                    else ("every checked dimension is identical, but "
                          f"{', '.join(unchecked)} "
                          + ("was" if len(unchecked) == 1 else "were")
                          + " never evaluated, so the difference cannot be "
                          "attributed to the rule alone")),
            isolates="the investment rule" if isolated else "",
            unchecked_dimensions=unchecked, dimension_results=tuple(results),
        )

    # A different evaluation period is the one difference that defeats even a
    # personal-outcome reading: two people measured over different markets have
    # no shared basis for the word "compared".
    if "evaluation_period" in differing:
        return ComparabilityVerdict(
            comparison_class=ComparisonClass.PERSONAL_OUTCOME,
            comparable=False, attribution_isolated=False,
            differing_dimensions=tuple(differing),
            detail=("the two were measured over different periods, so they were "
                    "exposed to different markets rather than to different rules"),
            isolates="",
            unchecked_dimensions=unchecked, dimension_results=tuple(results),
        )

    return ComparabilityVerdict(
        comparison_class=ComparisonClass.PERSONAL_OUTCOME,
        comparable=True, attribution_isolated=False,
        differing_dimensions=tuple(differing),
        detail=("comparable as lived outcomes; not as evidence about which rule "
                "is better"),
        isolates="",
        unchecked_dimensions=unchecked, dimension_results=tuple(results),
    )
