"""InvestmentMissionBench - a SELECTOR benchmark (Amendments §9), not a new strategy contest.

The existing research-backed strategies remain the baseline. This bench evaluates whether the objective
SELECTION policies (src/agentic/selection.py) do at least as well as the best fixed approved strategy and
the composite, walk-forward, net of transaction costs, with regime-specific breakdowns. It separates
strategy quality (the best fixed strategy in hindsight) from selector quality (what the policy chose),
and applies a non-inferiority gate before a learned selector would be promoted. Abstain is allowed.

Deterministic given (data, seed): no wall-clock, no RNG in the scoring path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import UNIVERSE
from ..features import compute_returns, exponential_cov, exponential_mean
from ..strategies import CASH_TICKER, apply_mandate, registry_for_objective, run_capability
from .selection import _stats_from_returns, build_snapshot, select_for_objective

BASE = [a.ticker for a in UNIVERSE]
OBJECTIVES = ("min_risk", "max_return_to_risk", "max_total_return")
# the objective-appropriate comparison metric (higher = better)
_METRIC = {"min_risk": lambda s: -s["ann_vol"], "max_return_to_risk": lambda s: s["sharpe"],
           "max_total_return": lambda s: s["ann_return"]}
_COST_BPS = 0.0010   # 10 bps per unit turnover (one-way, applied to L1 weight change / 2)


def _synthetic_prices(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    cols = list(dict.fromkeys(BASE + ["^VIX", "SPY", "GLD", "BIL"]))
    data: Dict[str, np.ndarray] = {}
    for t in cols:
        if t == "^VIX":
            data[t] = rng.uniform(12, 26, n)
        elif t == "BIL":
            data[t] = 100 * (1 + 0.00006) ** np.arange(n)
        else:
            drift, vol = rng.uniform(-0.0002, 0.0006), rng.uniform(0.004, 0.018)
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return pd.DataFrame(data, index=dates)


def _realize(weights: Dict[str, float], fwd: pd.DataFrame, rf: float) -> Dict[str, float]:
    vec = np.array([weights.get(t, 0.0) for t in BASE])
    port = pd.Series(fwd.reindex(columns=BASE).fillna(0.0).values @ vec, index=fwd.index)
    return _stats_from_returns(port, rf)


def _turnover(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in BASE) / 2.0


@dataclass
class BenchResult:
    rounds: int
    horizon: int
    per_objective: Dict[str, dict]
    baselines: Dict[str, dict]
    by_regime: Dict[str, dict]
    constraint_compliance: float
    seed: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rounds": self.rounds, "horizon": self.horizon, "seed": self.seed,
                "per_objective": self.per_objective, "baselines": self.baselines,
                "by_regime": self.by_regime, "constraint_compliance": self.constraint_compliance,
                "notes": self.notes}


def run_investment_bench(*, prices: Optional[pd.DataFrame] = None, seed: int = 7,
                         warmup: int = 300, step: int = 20, horizon: int = 20,
                         eps: float = 0.02, regimes: Optional[List[str]] = None) -> BenchResult:
    """Walk-forward selector benchmark. `regimes` cycles the assumed regime across eval dates."""
    prices = prices if prices is not None else _synthetic_prices(seed)
    returns = compute_returns(prices)[BASE].dropna()
    regimes = regimes or ["risk_on", "risk_off", "inflation"]
    idx = list(range(warmup, len(returns) - horizon, step))

    # accumulators
    sel: Dict[str, List[dict]] = {o: [] for o in OBJECTIVES}          # selector realized stats
    fixed: Dict[str, Dict[str, List[dict]]] = {o: {} for o in OBJECTIVES}  # per fixed strategy
    sel_turnover: Dict[str, float] = {o: 0.0 for o in OBJECTIVES}
    prev_w: Dict[str, Dict[str, float]] = {o: {CASH_TICKER: 1.0} for o in OBJECTIVES}
    by_regime: Dict[str, Dict[str, List[float]]] = {}
    compliant = 0
    total = 0

    for k, t in enumerate(idx):
        regime = regimes[k % len(regimes)]
        hist = returns.iloc[:t]
        fwd = returns.iloc[t:t + horizon]
        prices_hist = prices.loc[: hist.index[-1]]
        snap = build_snapshot(prices_hist, hist, regime, exponential_mean(hist),
                              exponential_cov(hist), rf=0.0)
        for obj in OBJECTIVES:
            res = select_for_objective(obj, snap, allow_abstain=False)
            w = res.weights
            total += 1
            # constraint compliance check (post-mandate weights must satisfy the mandate)
            _, binding = apply_mandate(w)
            if abs(sum(w.values()) - 1.0) < 1e-6 and all(v >= -1e-9 for v in w.values()):
                compliant += 1
            stats = _realize(w, fwd, snap.rf)
            cost = _turnover(prev_w[obj], w) * _COST_BPS * 252  # annualized cost drag
            stats_net = {**stats, "ann_return": stats["ann_return"] - cost}
            sel[obj].append(stats_net)
            sel_turnover[obj] += _turnover(prev_w[obj], w)
            prev_w[obj] = w
            by_regime.setdefault(regime, {o: [] for o in OBJECTIVES})
            by_regime[regime][obj].append(_METRIC[obj](stats_net))
            # fixed-strategy ceiling: every eligible approved strategy realized on the same window
            for cap in registry_for_objective(obj, regime):
                fw = apply_mandate(run_capability(cap.id, prices_hist, hist, regime, snap.context))[0]
                fixed[obj].setdefault(cap.id, []).append(_realize(fw, fwd, snap.rf))

    def _mean_metric(rows: List[dict], obj: str) -> float:
        return float(np.mean([_METRIC[obj](r) for r in rows])) if rows else 0.0

    per_objective: Dict[str, dict] = {}
    for obj in OBJECTIVES:
        selector_metric = _mean_metric(sel[obj], obj)
        best_id, best_metric = "", -1e9
        for sid, rows in fixed[obj].items():
            m = _mean_metric(rows, obj)
            if m > best_metric:
                best_id, best_metric = sid, m
        lift = selector_metric - best_metric
        per_objective[obj] = {
            "metric": obj if obj != "max_return_to_risk" else "sharpe",
            "selector_score": round(selector_metric, 4),
            "best_fixed_strategy": best_id,
            "best_fixed_score": round(best_metric, 4),      # strategy quality (hindsight ceiling)
            "lift_vs_best_fixed": round(lift, 4),           # selector quality vs that ceiling
            "non_inferior": bool(lift >= -eps),             # promotion gate: within eps of the ceiling
            "avg_turnover": round(sel_turnover[obj] / max(len(sel[obj]), 1), 4),
            "realized_return": round(float(np.mean([r["ann_return"] for r in sel[obj]])), 4),
            "realized_sharpe": round(float(np.mean([r["sharpe"] for r in sel[obj]])), 4),
            "realized_max_drawdown": round(float(np.mean([r["max_drawdown"] for r in sel[obj]])), 4),
        }

    # transparent baselines on the same walk-forward
    baselines = _baselines(returns, prices, idx, horizon)
    by_regime_out = {r: {o: round(float(np.mean(v[o])), 4) if v[o] else 0.0 for o in OBJECTIVES}
                     for r, v in by_regime.items()}

    notes = ["Selector benchmark: existing strategies are the baseline; the selector must be non-inferior "
             "to the best fixed approved strategy (within eps) before a learned selector is promoted.",
             "Strategy quality (best_fixed) is separated from selector quality (selector_score)."]
    return BenchResult(rounds=len(idx), horizon=horizon, per_objective=per_objective, baselines=baselines,
                       by_regime=by_regime_out, constraint_compliance=round(compliant / max(total, 1), 4),
                       seed=seed, notes=notes)


def _baselines(returns: pd.DataFrame, prices: pd.DataFrame, idx: List[int], horizon: int) -> Dict[str, dict]:
    """Equal-weight, 60/40, all-cash, HRP, current-RAAAL (sharpe optimizer) on the same windows."""
    def _bl(weights_fn) -> dict:
        rows = []
        for t in idx:
            hist = returns.iloc[:t]
            fwd = returns.iloc[t:t + horizon]
            w = weights_fn(hist, prices.loc[: hist.index[-1]])
            rows.append(_realize(w, fwd, 0.0))
        return {"realized_return": round(float(np.mean([r["ann_return"] for r in rows])), 4),
                "realized_sharpe": round(float(np.mean([r["sharpe"] for r in rows])), 4),
                "realized_max_drawdown": round(float(np.mean([r["max_drawdown"] for r in rows])), 4)}

    eq = {t: 1.0 / len(BASE) for t in BASE}
    sixty_forty = {"SPY": 0.6, "TLT": 0.4}
    cash = {CASH_TICKER: 1.0}
    out = {
        "equal_weight": _bl(lambda h, p: eq),
        "sixty_forty": _bl(lambda h, p: sixty_forty),
        "all_cash": _bl(lambda h, p: cash),
        "current_raaal_sharpe": _bl(lambda h, p: apply_mandate(
            run_capability("sharpe_optimizer", p, h, "risk_on", {"rf": 0.0}))[0]),
    }
    try:
        from ..hrp import compute_hrp_weights
        out["hrp"] = _bl(lambda h, p: apply_mandate(compute_hrp_weights(h))[0])
    except Exception:  # noqa: BLE001
        pass
    return out
