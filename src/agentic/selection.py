"""The Decision Planner: objective-specific SELECTION over the approved strategy registry.

The three user objectives are ranking policies, NOT new optimizers (Amendments §2, §4).
Each policy prunes the registry (objective / regime / promotion / mandate), scores the
eligible capabilities on held-out, regime-specific benchmark statistics, and selects one
strategy (or an approved ensemble). The recommended weights are then produced by that
strategy's own registered implementation and clipped to the hard mandate.

Objectives may legitimately coincide on a given snapshot — the difference lives in the
SCORING POLICY and the ranking, which are surfaced in EXPLAIN. Current/no-action is always
a comparison baseline, and abstain is allowed when expected benefit does not clear cost.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF, MANDATE_CONSTRAINTS, OBJECTIVES
from ..portfolio_utils import portfolio_metrics
from ..strategies import (
    CASH_TICKER,
    TICKERS,
    StrategyCapability,
    apply_mandate,
    registry_for_objective,
    run_capability,
)

_SQRT252 = float(np.sqrt(252))


# ---------------------------------------------------------------------------
# Evidence snapshot (built ONCE per cycle; shared by all three objectives)
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    prices: Optional[pd.DataFrame]
    returns: pd.DataFrame            # base-universe log returns
    regime: str
    mu: pd.Series
    cov: pd.DataFrame
    rf: float
    as_of: str
    snapshot_id: str
    context: Dict[str, object] = field(default_factory=dict)


def build_snapshot(
    prices: Optional[pd.DataFrame],
    returns: pd.DataFrame,
    regime: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: Optional[float] = None,
    context: Optional[Dict[str, object]] = None,
) -> Snapshot:
    base = returns.reindex(columns=TICKERS)
    as_of = str(base.index[-1].date()) if len(base) else "unknown"
    rf_val = float(rf) if rf is not None else DEFAULT_RF
    payload = f"{as_of}|{regime}|{'|'.join(TICKERS)}|{len(base)}|{rf_val:.8f}"
    snap_id = "snap_" + hashlib.sha256(payload.encode()).hexdigest()[:16]
    ctx = {"rf": rf_val}
    if context:
        ctx.update(context)
    return Snapshot(prices, base, regime, mu, cov, rf_val, as_of, snap_id, ctx)


# ---------------------------------------------------------------------------
# Held-out, regime-specific scoring statistics
# ---------------------------------------------------------------------------
def _port_returns(weights: Dict[str, float], returns: pd.DataFrame) -> pd.Series:
    vec = np.array([weights.get(t, 0.0) for t in TICKERS])
    mat = returns.reindex(columns=TICKERS).fillna(0.0).values
    return pd.Series(mat @ vec, index=returns.index)


def _stats_from_returns(port: pd.Series, rf: float) -> Dict[str, float]:
    ann_vol = float(port.std()) * _SQRT252 if not port.empty else 0.0
    # near-zero vol (e.g. an all-cash book) => Sharpe/Sortino undefined, not astronomically large
    if port.empty or ann_vol < 1e-8:
        return {"ann_return": float(port.mean()) * 252.0 if not port.empty else 0.0,
                "ann_vol": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "cvar95": 0.0}
    ann_return = float(port.mean()) * 252.0
    excess = ann_return - rf * 252.0
    sharpe = excess / ann_vol if ann_vol else 0.0
    downside = port[port < 0]
    dvol = float(downside.std()) * _SQRT252 if len(downside) > 1 else ann_vol
    sortino = excess / dvol if dvol else 0.0
    curve = (1.0 + port).cumprod()
    max_dd = float((curve / curve.cummax() - 1.0).min())
    tail = port[port <= port.quantile(0.05)]
    cvar95 = float(tail.mean()) if len(tail) else float(port.min())
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": max_dd, "cvar95": cvar95}


def held_out_stats(
    cap: StrategyCapability,
    snap: Snapshot,
    window: int = 252,
    split_frac: float = 0.6,
) -> Dict[str, float]:
    """Score a capability out-of-sample: fit weights on the in-sample slice, realize on the rest.

    A lightweight walk-forward proxy so scoring never look-aheads. The InvestmentMissionBench
    (P9) replaces this with full purged/embargoed regime-specific statistics injected via
    `benchmark_stats`.
    """
    rets = snap.returns
    n = len(rets)
    if n < 20:
        return _stats_from_returns(_port_returns(
            run_capability(cap.id, snap.prices, rets, snap.regime, snap.context), rets), snap.rf)
    sub = rets.iloc[max(0, n - window):]
    split = int(len(sub) * split_frac)
    if len(sub) - split < 10 or split < 10:
        # not enough for a held-out split — score current weights on the window
        w = run_capability(cap.id, snap.prices, sub, snap.regime, snap.context)
        return _stats_from_returns(_port_returns(w, sub), snap.rf)
    is_rets, oos_rets = sub.iloc[:split], sub.iloc[split:]
    prices_is = snap.prices
    if snap.prices is not None:
        prices_is = snap.prices.loc[: is_rets.index[-1]]
    weights = run_capability(cap.id, prices_is, is_rets, snap.regime, snap.context)
    return _stats_from_returns(_port_returns(weights, oos_rets), snap.rf)


# ---------------------------------------------------------------------------
# Objective selection policies (the only thing that differs per column)
# ---------------------------------------------------------------------------
class StrategySelectionPolicy:
    objective: str = ""
    label: str = ""

    def score(self, stats: Dict[str, float]) -> float:  # higher = better
        raise NotImplementedError

    def rationale(self, stats: Dict[str, float]) -> str:
        raise NotImplementedError


class MinRisk(StrategySelectionPolicy):
    objective, label = "min_risk", "Minimum Risk"

    def score(self, s: Dict[str, float]) -> float:
        return -(0.5 * s["ann_vol"] + 0.3 * abs(s["max_drawdown"]) + 0.2 * abs(s["cvar95"]) * _SQRT252)

    def rationale(self, s: Dict[str, float]) -> str:
        return f"vol {s['ann_vol']:.1%}, max drawdown {s['max_drawdown']:.1%}, CVaR {s['cvar95']:.2%}/day"


class MaxReturnToRisk(StrategySelectionPolicy):
    objective, label = "max_return_to_risk", "Maximum Return-to-Risk"

    def score(self, s: Dict[str, float]) -> float:
        return s["sharpe"] + 0.25 * s["sortino"]

    def rationale(self, s: Dict[str, float]) -> str:
        return f"Sharpe {s['sharpe']:.2f}, Sortino {s['sortino']:.2f} (held-out, this regime)"


class MaxTotalReturn(StrategySelectionPolicy):
    objective, label = "max_total_return", "Maximum Total Return"

    def score(self, s: Dict[str, float]) -> float:
        penalty = 0.5 * abs(s["max_drawdown"]) + (1.0 if s["max_drawdown"] < -0.5 else 0.0)
        return s["ann_return"] - penalty

    def rationale(self, s: Dict[str, float]) -> str:
        return f"CAGR {s['ann_return']:.1%} with max drawdown {s['max_drawdown']:.1%} (held-out, this regime)"


POLICIES: Dict[str, StrategySelectionPolicy] = {
    "min_risk": MinRisk(),
    "max_return_to_risk": MaxReturnToRisk(),
    "max_total_return": MaxTotalReturn(),
}


# ---------------------------------------------------------------------------
# Selection result (== the ObjectivePlan payload; pipeline wraps these)
# ---------------------------------------------------------------------------
@dataclass
class SelectionResult:
    objective: str
    selected_strategy_id: str
    alternative_strategy_ids: List[str]
    weights: Dict[str, float]
    regime: str
    evidence_snapshot_id: str
    benchmark_metrics: Dict[str, float]     # held-out stats of the selected strategy
    expected_tradeoffs: Dict[str, float]    # portfolio_metrics of the recommended weights
    binding_constraints: List[str]
    score_table: List[Dict[str, object]]    # ranked [{id, score, stats, promotion}]
    confidence: float
    warnings: List[str]
    abstained: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "objective": self.objective,
            "selected_strategy_id": self.selected_strategy_id,
            "alternative_strategy_ids": self.alternative_strategy_ids,
            "weights": self.weights,
            "regime": self.regime,
            "evidence_snapshot_id": self.evidence_snapshot_id,
            "benchmark_metrics": self.benchmark_metrics,
            "expected_tradeoffs": self.expected_tradeoffs,
            "binding_constraints": self.binding_constraints,
            "score_table": self.score_table,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "abstained": self.abstained,
        }


def _no_action_weights() -> Dict[str, float]:
    w = {t: 0.0 for t in TICKERS}
    w[CASH_TICKER] = 1.0
    return w


def select_for_objective(
    objective: str,
    snap: Snapshot,
    benchmark_stats: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    mandate: Optional[Dict[str, object]] = None,
    promotion: Tuple[str, ...] = ("approved",),
    allow_abstain: bool = True,
) -> SelectionResult:
    """Prune → score → select over the approved registry for one objective."""
    if objective not in POLICIES:
        raise ValueError(f"unknown objective: {objective}")
    policy = POLICIES[objective]
    eligible = registry_for_objective(objective, snap.regime, promotion)

    if not eligible:
        return SelectionResult(objective, "no_action", [], _no_action_weights(), snap.regime,
                               snap.snapshot_id, {}, {}, ["no eligible strategy"], [], 0.0,
                               ["No approved strategy is eligible for this objective in this regime."], True)

    ranked: List[Tuple[StrategyCapability, float, Dict[str, float]]] = []
    for cap in eligible:
        stats = None
        if benchmark_stats is not None:
            stats = benchmark_stats.get((cap.id, snap.regime)) or benchmark_stats.get((cap.id, "*"))
        if stats is None:
            stats = held_out_stats(cap, snap)
        ranked.append((cap, policy.score(stats), stats))
    ranked.sort(key=lambda r: r[1], reverse=True)

    top_cap, top_score, top_stats = ranked[0]
    alternatives = [c.id for c, _, _ in ranked[1:4]]
    score_table = [{"id": c.id, "family": c.family, "score": round(s, 4),
                    "promotion": c.promotion_status, "rationale": policy.rationale(st), "stats": st}
                   for c, s, st in ranked]

    # recommended weights come from the selected strategy's own implementation, then mandate-clipped
    raw = run_capability(top_cap.id, snap.prices, snap.returns, snap.regime, snap.context)
    weights, binding = apply_mandate(raw, mandate)
    expected = portfolio_metrics(weights, snap.mu, snap.cov, snap.rf)

    warnings: List[str] = []
    abstained = False
    if allow_abstain:
        # compare against current/no-action (all-cash) baseline for this objective
        cash_stats = _stats_from_returns(_port_returns(_no_action_weights(), snap.returns.tail(126)), snap.rf)
        if policy.score(top_stats) <= policy.score(cash_stats):
            warnings.append("Selected strategy does not beat holding cash on this objective — abstain suggested.")
            abstained = True

    # confidence from the score gap to the runner-up (bounded)
    gap = top_score - (ranked[1][1] if len(ranked) > 1 else top_score - 1.0)
    confidence = float(np.clip(0.55 + 0.5 * np.tanh(gap), 0.05, 0.99))

    return SelectionResult(
        objective=objective,
        selected_strategy_id=top_cap.id,
        alternative_strategy_ids=alternatives,
        weights=weights,
        regime=snap.regime,
        evidence_snapshot_id=snap.snapshot_id,
        benchmark_metrics=top_stats,
        expected_tradeoffs=expected,
        binding_constraints=binding,
        score_table=score_table,
        confidence=confidence,
        warnings=warnings,
        abstained=abstained,
    )


def select_all_objectives(
    snap: Snapshot,
    benchmark_stats: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    mandate: Optional[Dict[str, object]] = None,
    promotion: Tuple[str, ...] = ("approved",),
) -> Dict[str, SelectionResult]:
    """Run all three objective selection policies over ONE shared snapshot (no averaging across them)."""
    return {obj: select_for_objective(obj, snap, benchmark_stats, mandate, promotion)
            for obj in OBJECTIVES}
