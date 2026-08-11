"""One identity-bearing declaration of everything an RSU comparison depends on.

    A difference may be attributed to strategy only after every non-strategy
    dimension that can change the resulting sale quantity, proceeds, or
    reinvestment path has been checked and matched.

Before concentration sizing existed, matching vest flows was nearly enough: two
runs receiving identical dated value and differing only in disposition policy
were comparing dispositions. That stopped being true the moment a sale quantity
became a function of live portfolio state. Two runs can now agree on every flow,
every price and every date, disagree only about whether fractional shares may be
sold, and produce different sales — a difference that is not strategy and would
be reported as one.

**One object, not a list maintained in several places.** The registry of
dimensions already exists; what was missing was a single hashable declaration
covering the whole vest → sale → reinvest mechanism, so a run carries its
comparison identity rather than having it reassembled from scattered fields at
comparison time.

**A missing row is not a verdict.** Every requested benchmark produces a row,
including the ones that turned out to be incomparable. Filtering them before
rendering answers "which benchmarks agreed with us" while appearing to answer
"which benchmarks were run".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..runtime.base import canonical_hash

#: Bumped when a change would classify the same pair of runs differently.
#: Travels with every verdict, so a stored comparison says which rules produced
#: it rather than being re-judged by whatever is current.
PROFILE_VERSION = "comparison/rsu-profile@1"


class BenchmarkStatus(str, Enum):
    COMPARABLE = "COMPARABLE"
    """Every non-strategy dimension checked and matched."""

    COMPARABLE_WITH_UNCHECKED_DIMENSIONS = "COMPARABLE_WITH_UNCHECKED_DIMENSIONS"
    """Shown, and not isolated. Some required dimension was never established,
    so the difference cannot be attributed to the strategy — a weaker claim
    than COMPARABLE and a very different one from INCOMPARABLE."""

    PERSONAL_OUTCOME = "PERSONAL_OUTCOME"
    """Vest schedules or contributions differ. A real comparison of two lived
    outcomes that combines personal flows with strategy behaviour, and cannot
    claim either was isolated."""

    CONSTRAINT_EFFECT = "CONSTRAINT_EFFECT"
    INCOMPARABLE = "INCOMPARABLE"
    NOT_EVALUATED = "NOT_EVALUATED"
    """Requested and not run. Distinct from incomparable: one was judged, the
    other never happened."""


#: The four groups, and the fields in each. Declared as data so the equality
#: check cannot quietly omit one — `test_every_profile_field_is_load_bearing`
#: mutates each in turn and requires the verdict to change.
VEST_FLOW = ("grant_identity", "vest_dates", "delivered_values",
             "withholding_runtime", "corporate_action_runtime",
             "corporate_action_snapshot")

DISPOSITION = ("policy_kind", "blackout_policy", "execution_lag",
               "transaction_cost_model", "fractional_share_policy")

CONCENTRATION = ("employer_asset", "target_cap", "denominator_scope",
                 "valuation_session", "included_assets_policy",
                 "missing_price_policy", "rounding_policy")

ALLOCATION = ("allocation_policy", "target_assets", "target_weights",
              "funding_scope", "cash_reserve", "purchase_cost_model")

ENVIRONMENT = ("account_runtime", "tax_runtime", "calendar_runtime",
               "market_data_runtime", "evaluation_period")

ALL_FIELDS = VEST_FLOW + DISPOSITION + CONCENTRATION + ALLOCATION + ENVIRONMENT

#: Which group each field belongs to, for naming what differed.
GROUP_OF: Dict[str, str] = {
    **{name: "vest_flow" for name in VEST_FLOW},
    **{name: "disposition" for name in DISPOSITION},
    **{name: "concentration" for name in CONCENTRATION},
    **{name: "allocation" for name in ALLOCATION},
    **{name: "environment" for name in ENVIRONMENT},
}

#: Fields a strategy comparison is *about*. Everything else must match.
STRATEGY_FIELDS = frozenset({"policy_kind", "allocation_policy",
                             "target_assets", "target_weights", "target_cap"})

#: Fields whose difference makes the comparison a personal outcome rather than a
#: strategy claim: they describe what the person received, not what they did.
PERSONAL_FIELDS = frozenset({"grant_identity", "vest_dates",
                             "delivered_values"})


@dataclass(frozen=True)
class RSUComparisonProfile:
    """Everything that can change the sale quantity, proceeds or reinvestment.

    Unset fields are `None`, which compares as *not evaluated* rather than as
    equal. Two runs that both left the corporate-action runtime unpinned have
    not agreed about corporate actions; they have both declined to say.
    """

    # vest flow
    grant_identity: Optional[str] = None
    vest_dates: Optional[Sequence[str]] = None
    delivered_values: Optional[Sequence[float]] = None
    withholding_runtime: Optional[str] = None
    corporate_action_runtime: Optional[str] = None
    corporate_action_snapshot: Optional[str] = None
    """Which realized action history the run received.

    Beside the runtime, not instead of it. Two runs can share an interpretation
    policy and be handed different histories — one knowing about a split the
    other does not — and matching only the policy would repeat the market-data
    snapshot defect exactly."""

    # disposition
    policy_kind: Optional[str] = None
    blackout_policy: Optional[Sequence[tuple]] = None
    execution_lag: Optional[int] = None
    transaction_cost_model: Optional[str] = None
    fractional_share_policy: Optional[str] = None

    # concentration
    employer_asset: Optional[str] = None
    target_cap: Optional[float] = None
    denominator_scope: Optional[Sequence[str]] = None
    valuation_session: Optional[str] = None
    included_assets_policy: Optional[str] = None
    missing_price_policy: Optional[str] = None
    rounding_policy: Optional[str] = None

    # allocation
    allocation_policy: Optional[str] = None
    target_assets: Optional[Sequence[str]] = None
    target_weights: Optional[Mapping[str, float]] = None
    funding_scope: Optional[str] = None
    cash_reserve: Optional[float] = None
    purchase_cost_model: Optional[str] = None

    # environment
    account_runtime: Optional[str] = None
    tax_runtime: Optional[str] = None
    calendar_runtime: Optional[str] = None
    market_data_runtime: Optional[str] = None
    evaluation_period: Optional[tuple] = None

    profile_version: str = PROFILE_VERSION

    def dimension_map(self) -> Dict[str, Any]:
        """Every field, comparably. Ordered collections are normalised to
        tuples so a list and a tuple of the same values do not read as a
        difference in the mechanism."""
        values: Dict[str, Any] = {}
        for name in ALL_FIELDS:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = tuple(sorted(value.items()))
            elif isinstance(value, (list, tuple)):
                value = tuple(value)
            values[name] = value
        return values

    @property
    def compatibility_hash(self) -> str:
        """One identity for the whole mechanism.

        Persisted on every RSU run, so a stored comparison can be re-checked
        against the profile that produced it rather than against whatever the
        code now assembles.
        """
        return canonical_hash({"version": self.profile_version,
                               **{k: str(v) for k, v
                                  in self.dimension_map().items()}})

    def unevaluated(self) -> Sequence[str]:
        return tuple(name for name, value in self.dimension_map().items()
                     if value is None)

    def to_json(self) -> Dict[str, Any]:
        return {"profile_version": self.profile_version,
                "compatibility_hash": self.compatibility_hash,
                "fields": {k: v for k, v in self.dimension_map().items()},
                "unevaluated": list(self.unevaluated())}


@dataclass(frozen=True)
class BenchmarkVerdict:
    """One requested benchmark's row. Persisted whatever the outcome."""

    benchmark_id: str
    status: BenchmarkStatus
    requested: bool = True
    classifier_version: str = PROFILE_VERSION
    comparison_profile_hash: str = ""
    benchmark_flow_mode: str = ""
    matched_dimensions: Sequence[str] = ()
    differing_dimensions: Sequence[str] = ()
    unchecked_dimensions: Sequence[str] = ()
    isolates: str = ""
    reason: str = ""

    @property
    def attribution_isolated(self) -> bool:
        """Never true while a required dimension went unchecked.

        A comparison may still be shown. It may not claim the strategy was the
        cause when something that could also have caused it was never looked at.
        """
        return (self.status is BenchmarkStatus.COMPARABLE
                and not self.unchecked_dimensions)

    def to_json(self) -> Dict[str, Any]:
        return {"benchmark_id": self.benchmark_id, "requested": self.requested,
                "status": self.status.value,
                "classifier_version": self.classifier_version,
                "comparison_profile_hash": self.comparison_profile_hash,
                "benchmark_flow_mode": self.benchmark_flow_mode,
                "matched_dimensions": list(self.matched_dimensions),
                "differing_dimensions": list(self.differing_dimensions),
                "unchecked_dimensions": list(self.unchecked_dimensions),
                "attribution_isolated": self.attribution_isolated,
                "isolates": self.isolates, "reason": self.reason}


def classify(strategy: RSUComparisonProfile, benchmark: RSUComparisonProfile,
             *, benchmark_id: str, isolating: Sequence[str] = (),
             flow_mode: str = "") -> BenchmarkVerdict:
    """What a difference between these two runs would mean.

    `isolating` names the dimensions the comparison is *about* — the strategy
    axis being tested. Everything else must match. A comparison of "fixed sale
    versus concentration-targeted sale" isolates the concentration policy, and
    every mechanic around it must then be equal.

    Neither result is an input. A verdict computed from outcomes would be a
    verdict about which answer is convenient.
    """
    left, right = strategy.dimension_map(), benchmark.dimension_map()
    isolated = set(isolating)

    matched: List[str] = []
    differing: List[str] = []
    unchecked: List[str] = []

    for name in ALL_FIELDS:
        a, b = left[name], right[name]
        if a is None or b is None:
            # Absent on either side is NOT_EVALUATED, never "equal". Two runs
            # that both declined to pin the corporate-action runtime have not
            # agreed about corporate actions.
            unchecked.append(name)
        elif a == b:
            matched.append(name)
        else:
            differing.append(name)

    unexpected = [name for name in differing if name not in isolated]
    personal = [name for name in unexpected if name in PERSONAL_FIELDS]
    required_unchecked = [name for name in unchecked if name not in isolated]

    profile_hash = strategy.compatibility_hash

    if personal:
        return BenchmarkVerdict(
            benchmark_id=benchmark_id, status=BenchmarkStatus.PERSONAL_OUTCOME,
            comparison_profile_hash=profile_hash, benchmark_flow_mode=flow_mode,
            matched_dimensions=tuple(matched),
            differing_dimensions=tuple(differing),
            unchecked_dimensions=tuple(required_unchecked),
            reason=("vest schedules or delivered values differ, so this "
                    "combines personal flows with strategy behaviour and "
                    "isolates neither: "
                    + ", ".join(sorted(personal))))

    if unexpected:
        return BenchmarkVerdict(
            benchmark_id=benchmark_id, status=BenchmarkStatus.INCOMPARABLE,
            comparison_profile_hash=profile_hash, benchmark_flow_mode=flow_mode,
            matched_dimensions=tuple(matched),
            differing_dimensions=tuple(differing),
            unchecked_dimensions=tuple(required_unchecked),
            reason=("these differ outside the strategy being tested, so a "
                    "difference in outcome is not the strategy: "
                    + ", ".join(f"{GROUP_OF[n]}.{n}" for n in sorted(unexpected))))

    if required_unchecked:
        return BenchmarkVerdict(
            benchmark_id=benchmark_id,
            status=BenchmarkStatus.COMPARABLE_WITH_UNCHECKED_DIMENSIONS,
            comparison_profile_hash=profile_hash, benchmark_flow_mode=flow_mode,
            matched_dimensions=tuple(matched),
            differing_dimensions=tuple(differing),
            unchecked_dimensions=tuple(required_unchecked),
            reason=("This comparison is shown, but the strategy effect is not "
                    "isolated because the following dimensions were not "
                    "established: "
                    + ", ".join(f"{GROUP_OF[n]}.{n}"
                                for n in sorted(required_unchecked))))

    return BenchmarkVerdict(
        benchmark_id=benchmark_id, status=BenchmarkStatus.COMPARABLE,
        comparison_profile_hash=profile_hash, benchmark_flow_mode=flow_mode,
        matched_dimensions=tuple(matched),
        differing_dimensions=tuple(differing),
        isolates=", ".join(sorted(isolated)) if isolated else "",
        reason=("every non-strategy dimension was checked and matched"
                if isolated else "these runs are identical in every dimension"))


def evaluate(strategy: RSUComparisonProfile,
             requested: Mapping[str, Any]) -> List[BenchmarkVerdict]:
    """A verdict row for every requested benchmark, whatever the outcome.

    Nothing is filtered. A benchmark that could not be built, could not be run,
    or turned out incomparable still produces a row saying so — otherwise the
    displayed set answers "which benchmarks agreed with us" while appearing to
    answer "which benchmarks were requested".
    """
    rows: List[BenchmarkVerdict] = []
    for benchmark_id, spec in requested.items():
        profile = spec.get("profile") if isinstance(spec, Mapping) else None
        if profile is None:
            rows.append(BenchmarkVerdict(
                benchmark_id=benchmark_id,
                status=BenchmarkStatus.NOT_EVALUATED,
                comparison_profile_hash=strategy.compatibility_hash,
                reason=(spec.get("reason") if isinstance(spec, Mapping) else "")
                or "requested and not run"))
            continue
        rows.append(classify(
            strategy, profile, benchmark_id=benchmark_id,
            isolating=spec.get("isolating", ()),
            flow_mode=spec.get("flow_mode", "")))
    return rows
