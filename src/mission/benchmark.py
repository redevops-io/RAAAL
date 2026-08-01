"""Benchmarks that receive the same money on the same days.

Comparing "I contribute $2,000/month to my plan" against "buy and hold SPY"
compares two things that differ in *both* strategy and contribution schedule.
Whatever difference comes out cannot be attributed to either, which makes the
comparison invalid in the same way two methodologies with different output
contracts are — and the platform already refuses that comparison.

So flow matching is a **comparability property**, not a rendering convenience,
and a benchmark that cannot receive the schedule is reported as incomparable
rather than quietly dropped. A comparison set that silently excludes what does
not fit is a curated argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .accounting import CashFlow, CashPolicy, Order
from .recommendation import assess as assess_recommendation
from .simulate import MissionResult, simulate


@dataclass(frozen=True)
class FlowMismatch:
    """Why a benchmark cannot be compared against this Mission."""

    benchmark: str
    field: str
    why: str

    def to_json(self) -> Dict[str, Any]:
        return {"benchmark": self.benchmark, "field": self.field, "why": self.why}


@dataclass(frozen=True)
class BenchmarkComparison:
    """One benchmark's result, or the reason there isn't one."""

    name: str
    description: str
    result: Optional[MissionResult] = None
    mismatch: Optional[FlowMismatch] = None

    @property
    def comparable(self) -> bool:
        return self.result is not None and self.mismatch is None

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "comparable": self.comparable,
            "result": self.result.to_json() if self.result else None,
            "mismatch": self.mismatch.to_json() if self.mismatch else None,
        }


def buy_and_hold(tickers: Sequence[str], *, weights: Optional[Dict[str, float]] = None):
    """Invest every arriving dollar across `tickers`, and never sell.

    This is the honest flow-matched analogue of "buy and hold": the schedule is
    the Mission's, so the only difference left is what the money bought.
    """
    weights = weights or {t: 1.0 / len(tickers) for t in tickers}
    total = sum(weights.values())
    share = {t: w / total for t, w in weights.items()}

    def program(session, visible, holdings, cash):
        if cash <= 0:
            return ()
        return [
            Order(date=session, ticker=t, notional=cash * s,
                  reason="flow-matched buy and hold")
            for t, s in share.items() if s > 0
        ]

    return program


def hold_cash():
    """Contribute and never invest. The comparison nobody runs and everybody needs."""
    def program(session, visible, holdings, cash):
        return ()
    return program


def compare(
    prices: pd.DataFrame,
    *,
    flows: Sequence[CashFlow],
    benchmarks: Sequence[Dict[str, Any]],
    cash_policy: CashPolicy,
    execution_lag: int = 1,
    cost_bps: float = 10.0,
    periods_per_year: int = 252,
) -> List[BenchmarkComparison]:
    """Run every benchmark on identical flows, costs, calendar and lag.

    Deliberately returns the set in declaration order and never sorted by
    outcome. Ordering a comparison by result turns a symmetric set of facts into
    a ranking, and a ranking presented by the platform is a recommendation the
    platform did not intend to make.
    """
    out: List[BenchmarkComparison] = []
    for spec in benchmarks:
        name = spec["name"]
        missing = [t for t in spec.get("tickers", ()) if t not in prices.columns]
        if missing:
            out.append(BenchmarkComparison(
                name=name,
                description=spec.get("description", ""),
                mismatch=FlowMismatch(
                    benchmark=name, field="price_coverage",
                    why=(f"No price history for {', '.join(missing)} over this "
                         "period, so this benchmark cannot receive the same "
                         "contributions on the same days."),
                ),
            ))
            continue

        result = simulate(
            prices, flows=flows, program=spec["program"],
            cash_policy=cash_policy, execution_lag=execution_lag,
            cost_bps=cost_bps, periods_per_year=periods_per_year,
        )
        out.append(BenchmarkComparison(
            name=name, description=spec.get("description", ""), result=result,
        ))
    return out


def comparison_payload(
    mission: MissionResult,
    benchmarks: Sequence[BenchmarkComparison],
    *,
    rendered_text: str = "",
    declared_order: Optional[Sequence[str]] = None,
    user_originated_rule: Optional[bool] = None,
    platform_generated_action: Optional[bool] = None,
    portfolio_selection_performed: Optional[bool] = None,
) -> Dict[str, Any]:
    """The comparison as data, with no conclusion attached.

    `is_recommendation` is **derived** rather than declared. An earlier version of
    this function emitted the literal `False`, which asserted the platform's own
    compliance and could not be wrong — the same defect as a methodology
    declaring a rule the executor never enforces.

    The strongest of the nine checks is the ordering one, because it cannot be
    satisfied by wording: if the payload arrives sorted by outcome it is a
    ranking, whatever the accompanying prose says.
    """
    incomparable = [b for b in benchmarks if not b.comparable]
    names = [b.name for b in benchmarks]

    verdict = assess_recommendation(
        benchmarks=benchmarks,
        rendered_text=rendered_text,
        declared_order=list(declared_order) if declared_order is not None else names,
        payload_order=names,
        ordering_metric=[b.result.money_weighted if b.result else None
                         for b in benchmarks],
        user_originated_rule=user_originated_rule,
        platform_generated_action=platform_generated_action,
        portfolio_selection_performed=portfolio_selection_performed,
    )

    return {
        "mission": mission.to_json(),
        "benchmarks": [b.to_json() for b in benchmarks],
        "recommendation_assessment": verdict.to_json(),
        "is_recommendation": verdict.is_recommendation,
        "incomparable_count": len(incomparable),
        "note": (
            "Every comparable benchmark received identical contributions on "
            "identical days, under identical costs, execution lag and calendar, "
            "so the only difference between them is what the money bought. The "
            "set is returned in declaration order; ranking it is the reader's to do."
        ),
    }
