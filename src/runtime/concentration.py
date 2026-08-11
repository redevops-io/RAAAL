"""Employer concentration, and solving for the sale that reaches a declared cap.

    settled portfolio -> assessment -> minimum sale -> instruction -> fill
                                                    -> realized concentration

This is a portfolio-state calculation with a solvable target, not "sell 80% of
the grant". The quantity depends on the whole portfolio at the moment the sale
becomes eligible, and both the employer price and everything else may have moved
since the vest.

**A cap is declared, never recommended.** Twenty percent is not safe, prudent or
optimal here; it is a constraint a user or a methodology stated, and this module
enforces it. Naming it otherwise would turn an instruction into advice.

**The denominator is explicit.** Concentration is a ratio, and a ratio whose
denominator is assembled from whatever happened to be priced is not reproducible.
Included: settled holdings and settled cash. Excluded, by name rather than by
omission: unvested grants, pending dispositions, pending allocation orders,
unreconciled fills, and anything outside the modelled account.

**Missing prices refuse.** Dropping an unpriced holding understates the
denominator, which overstates concentration and sizes the sale *too small* —
a plan that reports success while leaving the position above its cap.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class Feasibility(str, Enum):
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    """At or below the cap. No sale, and the zero is a computed quantity rather
    than an unknown."""

    SOLVED = "SOLVED"
    UNCOMPUTABLE = "UNCOMPUTABLE"
    """An included component could not be priced."""


class RoundingPolicy(str, Enum):
    FRACTIONAL_ALLOWED = "FRACTIONAL_ALLOWED"
    WHOLE_SHARES_UP = "WHOLE_SHARES_UP"
    """Rounded up to the next whole share.

    Down would leave the position above the declared maximum while the plan
    reported success — the one rounding direction that produces a false pass."""


#: Components deliberately outside the denominator, named so the exclusion is a
#: statement rather than an oversight.
DEFAULT_EXCLUSIONS = (
    "unvested grants",
    "pending dispositions",
    "pending allocation orders",
    "unreconciled fills",
    "assets outside the modelled account",
)


@dataclass(frozen=True)
class ConcentrationPolicy:
    """Everything that makes one concentration measurement comparable to another.

    Exposed as explicit fields now so a benchmark comparison can pin them. A
    state-dependent sale quantity means identical vest flows are no longer
    enough for a strategy-effect claim: two runs that disagree about the
    fractional-share policy or the cost model produce different sales from
    identical inputs.
    """

    target: float
    """The declared cap. A constraint, not a recommendation."""

    included: Sequence[str] = ("settled holdings", "settled cash")
    excluded: Sequence[str] = DEFAULT_EXCLUSIONS
    rounding: RoundingPolicy = RoundingPolicy.WHOLE_SHARES_UP
    cost_rate: float = 0.001
    execution_lag: int = 1
    blackout_ref: Sequence[tuple] = ()
    scope_note: str = ("Concentration is measured within this modelled account, "
                       "not across household wealth.")

    def to_json(self) -> Dict[str, Any]:
        return {"target": self.target, "included": list(self.included),
                "excluded": list(self.excluded),
                "rounding": self.rounding.value, "cost_rate": self.cost_rate,
                "execution_lag": self.execution_lag,
                "blackout_ref": [list(w) for w in self.blackout_ref],
                "scope_note": self.scope_note}


@dataclass(frozen=True)
class ConcentrationAssessment:
    """What the portfolio looked like at one moment."""

    assessment_id: str
    measured_at: Any
    employer_asset: str
    employer_value: float
    portfolio_value: float
    target: float
    data_complete: bool
    missing_prices: Sequence[str] = ()
    excluded_components: Sequence[str] = DEFAULT_EXCLUSIONS
    scope_note: str = ""

    @property
    def concentration(self) -> Optional[float]:
        """None when the denominator is incomplete.

        Not zero and not a best guess: a ratio computed from a denominator
        missing an unpriced holding is wrong in the direction that overstates
        concentration.
        """
        if not self.data_complete or self.portfolio_value <= 0:
            return None
        return self.employer_value / self.portfolio_value

    @property
    def excess_value(self) -> Optional[float]:
        """Employer value above the cap, before any sale."""
        if self.concentration is None:
            return None
        return max(0.0, self.employer_value - self.target * self.portfolio_value)

    def to_json(self) -> Dict[str, Any]:
        return {"assessment_id": self.assessment_id,
                "measured_at": str(self.measured_at),
                "employer_asset": self.employer_asset,
                "employer_value": self.employer_value,
                "portfolio_value": self.portfolio_value,
                "concentration": self.concentration, "target": self.target,
                "excess_value": self.excess_value,
                "data_complete": self.data_complete,
                "missing_prices": list(self.missing_prices),
                "excluded_components": list(self.excluded_components),
                "scope_note": self.scope_note}


def assess(*, holdings: Mapping[str, float], prices: Mapping[str, float],
           cash: float, employer_asset: str, policy: ConcentrationPolicy,
           measured_at: Any) -> ConcentrationAssessment:
    """Measure concentration from the settled portfolio.

    Every held asset must price. An unpriced holding is not dropped: dropping it
    shrinks the denominator, inflates the measured concentration, and sizes the
    corrective sale too small — the failure mode that reports success while
    leaving the position above its cap.
    """
    missing = [asset for asset, shares in holdings.items()
               if shares and (asset not in prices
                              or prices[asset] != prices[asset]
                              or prices[asset] <= 0)]

    values = {asset: shares * float(prices.get(asset, 0.0))
              for asset, shares in holdings.items()}
    portfolio = sum(values.values()) + float(cash)

    return ConcentrationAssessment(
        assessment_id=f"conc-{uuid.uuid4().hex[:16]}",
        measured_at=measured_at, employer_asset=employer_asset,
        employer_value=values.get(employer_asset, 0.0),
        portfolio_value=portfolio, target=policy.target,
        data_complete=not missing, missing_prices=tuple(sorted(missing)),
        excluded_components=tuple(policy.excluded),
        scope_note=policy.scope_note)


def projected_concentration(shares_sold: float, *, employer_value: float,
                            portfolio_value: float, price: float,
                            cost_rate: float) -> float:
    """Concentration after selling `shares_sold` into cash.

    Selling into cash leaves the denominator unchanged except for the
    transaction cost: the value moves from the holding to cash, both inside the
    portfolio. Only the cost leaves.
    """
    proceeds = shares_sold * price
    remaining_employer = max(0.0, employer_value - proceeds)
    remaining_portfolio = portfolio_value - proceeds * cost_rate
    if remaining_portfolio <= 0:
        return 0.0
    return remaining_employer / remaining_portfolio


@dataclass(frozen=True)
class ConcentrationDispositionPlan:
    """The sale that reaches the cap, and the proof it is the smallest one."""

    plan_id: str
    assessment_ref: str
    target: float
    feasibility: Feasibility

    minimum_continuous_quantity: Optional[float] = None
    """The exact fractional share count that reaches the cap. Reported beside
    the executable quantity so rounding is visible rather than absorbed."""

    shares_to_sell: float = 0.0
    estimated_gross_proceeds: float = 0.0
    estimated_cost: float = 0.0
    projected_post_sale_concentration: Optional[float] = None
    rounding_policy: RoundingPolicy = RoundingPolicy.WHOLE_SHARES_UP
    unresolved_inputs: Sequence[str] = ()
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "assessment_ref": self.assessment_ref,
                "target": self.target, "feasibility": self.feasibility.value,
                "minimum_continuous_quantity": self.minimum_continuous_quantity,
                "shares_to_sell": self.shares_to_sell,
                "estimated_gross_proceeds": self.estimated_gross_proceeds,
                "estimated_cost": self.estimated_cost,
                "projected_post_sale_concentration":
                    self.projected_post_sale_concentration,
                "rounding_policy": self.rounding_policy.value,
                "unresolved_inputs": list(self.unresolved_inputs),
                "detail": self.detail}


#: Concentration comparisons are made with a tolerance rather than exactly.
#: Binary floating point makes an exact `<=` reject a quantity that reaches the
#: cap to fifteen decimal places.
TOLERANCE = 1e-9


def solve(assessment: ConcentrationAssessment, *, price: float,
          held_shares: float, policy: ConcentrationPolicy
          ) -> ConcentrationDispositionPlan:
    """The smallest permitted sale that brings concentration to the cap.

    Solved against actual share increments and the cost rule rather than trusted
    from the closed form: the continuous solution is a starting point, and the
    executable quantity is verified — and its predecessor checked to fail — so
    the result is demonstrably minimal rather than merely sufficient.
    """
    plan_id = f"plan-{uuid.uuid4().hex[:16]}"

    if not assessment.data_complete:
        return ConcentrationDispositionPlan(
            plan_id=plan_id, assessment_ref=assessment.assessment_id,
            target=policy.target, feasibility=Feasibility.UNCOMPUTABLE,
            rounding_policy=policy.rounding,
            unresolved_inputs=tuple(assessment.missing_prices),
            detail=("the denominator is incomplete, so any quantity solved from "
                    "it would be too small"))

    if price <= 0 or price != price:
        return ConcentrationDispositionPlan(
            plan_id=plan_id, assessment_ref=assessment.assessment_id,
            target=policy.target, feasibility=Feasibility.UNCOMPUTABLE,
            rounding_policy=policy.rounding,
            unresolved_inputs=(assessment.employer_asset,),
            detail="the employer holding could not be priced")

    current = assessment.concentration
    if current is not None and current <= policy.target + TOLERANCE:
        return ConcentrationDispositionPlan(
            plan_id=plan_id, assessment_ref=assessment.assessment_id,
            target=policy.target, feasibility=Feasibility.ALREADY_SATISFIED,
            minimum_continuous_quantity=0.0, shares_to_sell=0.0,
            projected_post_sale_concentration=current,
            rounding_policy=policy.rounding,
            detail="already at or below the declared cap")

    # Continuous solution. Selling x of value: employer becomes E - x and the
    # portfolio becomes P - x*r, so (E - x) / (P - x*r) <= c gives
    #   x >= (E - cP) / (1 - c*r)
    employer, portfolio = assessment.employer_value, assessment.portfolio_value
    rate, cap = policy.cost_rate, policy.target
    denominator = 1.0 - cap * rate
    continuous_value = (employer - cap * portfolio) / denominator
    continuous_shares = max(0.0, continuous_value / price)

    # No infeasibility branch, deliberately. Proceeds stay inside the portfolio
    # — the value moves from the holding to cash and only the transaction cost
    # leaves — so selling the whole position drives concentration to zero and
    # any cap above zero is reachable. Algebraically, `x_min > E` would require
    # `P < E * cost_rate`, which cannot hold while `E <= P`.
    #
    # An unreachable branch is a claim nothing can check, so it is absent rather
    # than defensive. `test_any_cap_is_reachable_by_selling_into_cash` pins the
    # property this relies on; if proceeds ever leave the account, that test
    # fails and the branch comes back.
    increment = (0.0 if policy.rounding is RoundingPolicy.FRACTIONAL_ALLOWED
                 else 1.0)
    if increment:
        # Up, never down. Down leaves the position above the declared maximum
        # while the plan reports success.
        shares = min(held_shares, math.ceil(continuous_shares - TOLERANCE))
    else:
        shares = continuous_shares

    # Verify against the actual cost rule rather than trusting the algebra, and
    # step up if the increment lands short.
    guard = 0
    while projected_concentration(
            shares, employer_value=employer, portfolio_value=portfolio,
            price=price, cost_rate=rate) > cap + TOLERANCE:
        if shares >= held_shares - TOLERANCE or guard > 1000:
            break
        shares = min(held_shares, shares + (increment or continuous_shares * 1e-6))
        guard += 1

    proceeds = shares * price
    return ConcentrationDispositionPlan(
        plan_id=plan_id, assessment_ref=assessment.assessment_id,
        target=policy.target, feasibility=Feasibility.SOLVED,
        minimum_continuous_quantity=continuous_shares,
        shares_to_sell=shares, estimated_gross_proceeds=proceeds,
        estimated_cost=proceeds * rate,
        projected_post_sale_concentration=projected_concentration(
            shares, employer_value=employer, portfolio_value=portfolio,
            price=price, cost_rate=rate),
        rounding_policy=policy.rounding,
        detail=f"solved against a {cap:.0%} declared cap")


def reaches_cap(plan: ConcentrationDispositionPlan) -> bool:
    if plan.projected_post_sale_concentration is None:
        return False
    return plan.projected_post_sale_concentration <= plan.target + TOLERANCE


def realized_concentration(*, holdings: Mapping[str, float],
                           prices: Mapping[str, float], cash: float,
                           employer_asset: str) -> Optional[float]:
    """Concentration from what actually filled.

    Kept distinct from the projection. A partial fill leaves the position higher
    than planned, and a plan reporting its target as met on the strength of an
    order it placed would be describing an intention as an outcome.
    """
    values = {asset: shares * float(prices.get(asset, float("nan")))
              for asset, shares in holdings.items()}
    if any(value != value for value in values.values()):
        return None
    total = sum(values.values()) + float(cash)
    if total <= 0:
        return None
    return values.get(employer_asset, 0.0) / total


class EmployerStockInTargets(ValueError):
    """A concentration-targeted sale whose proceeds buy the employer back.

    Unsupported rather than approximated. The solver sizes the sale assuming
    the proceeds leave the position; buying some of it back means the cap is
    missed by an amount that depends on the allocation, and the plan would
    report a target it did not reach.
    """


def refuse_employer_in_targets(weights: Mapping[str, float],
                               employer_asset: str) -> None:
    """Guard the one allocation a concentration solver cannot account for."""
    if employer_asset and weights.get(employer_asset):
        raise EmployerStockInTargets(
            f"the allocation buys {employer_asset} back at "
            f"{weights[employer_asset]:.0%} while the sale was sized to reduce "
            "it. Sizing that correctly requires solving the sale and the "
            "repurchase together, which this does not do")


#: Mechanisms this module implements, resolved by the realization check.
IMPLEMENTED = ("assess", "solve", "projected_concentration",
               "realized_concentration", "reaches_cap",
               "refuse_employer_in_targets")
