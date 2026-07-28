"""Discovery Runtime signal adapters: portfolio/market observations -> agentic_os Signals.

Turns one evidence snapshot (regime, market stats, portfolio drift/risk, per-strategy trailing excess,
data freshness) into normalized `Signal`s the deterministic detectors read. Pure numeric mapping - no
LLM, no strategy logic. The detectors + runtime live in detectors.py / discovery_runtime.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agentic_os.discovery import Signal

from ..config import UNIVERSE

PORTFOLIO_SUBJECT = "portfolio.raaal-demo"


@dataclass
class PortfolioObservation:
    """One cycle's numeric observation of the market + paper portfolio (all optional; defaults are calm)."""
    regime: str = "risk_on"
    prev_regime: Optional[str] = None
    vix: float = 16.0
    avg_correlation: float = 0.3
    breadth: float = 0.55                 # fraction of assets in uptrend
    drawdown: float = 0.0                 # <= 0
    expected_shortfall: float = -0.02     # daily CVaR (<= 0)
    cash_weight: float = 0.10
    gross_leverage: float = 1.0
    data_age_days: float = 0.0
    rule_ml_agreement: float = 1.0        # 1.0 = rule and ML regimes agree, 0.0 = diverge
    portfolio_drift: float = 0.0          # L1/2 distance of current holdings vs the objective target
    hhi: float = 0.15                     # Herfindahl concentration of weights (0..1)
    max_weight: float = 0.25
    expected_benefit: float = 0.0         # expected bp improvement from rebalancing
    rebalance_cost: float = 0.0           # expected bp cost of rebalancing
    strategy_excess: Dict[str, float] = field(default_factory=dict)  # strategy_id -> trailing excess vs benchmark
    subject: str = PORTFOLIO_SUBJECT


def _hhi(weights: Dict[str, float]) -> float:
    return float(sum(float(w) ** 2 for w in weights.values()))


def observation_from(compare: dict, portfolio_state: dict, *,
                     vix: float = 16.0, avg_correlation: float = 0.3, breadth: float = 0.55,
                     data_age_days: float = 0.0, rule_ml_agreement: float = 1.0,
                     strategy_excess: Optional[Dict[str, float]] = None) -> PortfolioObservation:
    """Assemble an observation from an ObjectiveCompareResult.to_dict() + the portfolio state."""
    holdings = {k: float(v) for k, v in (portfolio_state.get("holdings", {}) or {}).items()}
    # objective target used for drift = the return-to-risk plan (the balanced default)
    plans = compare.get("plans", {})
    target = (plans.get("max_return_to_risk", {}) or {}).get("weights", {}) or {}
    drift = sum(abs(float(target.get(t, 0.0)) - holdings.get(t, 0.0)) for t in set(holdings) | set(target)) / 2.0
    bm = (plans.get("min_risk", {}) or {}).get("benchmark_metrics", {}) or {}
    return PortfolioObservation(
        regime=compare.get("regime", "risk_on"),
        vix=vix, avg_correlation=avg_correlation, breadth=breadth,
        drawdown=float(bm.get("max_drawdown", 0.0)),
        expected_shortfall=float(bm.get("cvar95", -0.02)),
        cash_weight=holdings.get("BIL", 0.0),
        data_age_days=data_age_days, rule_ml_agreement=rule_ml_agreement,
        portfolio_drift=drift, hhi=_hhi(holdings) if holdings else 0.15,
        max_weight=max(holdings.values()) if holdings else 0.25,
        strategy_excess=strategy_excess or {},
    )


def build_signals(obs: PortfolioObservation) -> List[Signal]:
    """Emit the deterministic Signal set for one observation."""
    subj = obs.subject
    sigs: List[Signal] = [
        Signal("raaal", subj, "vix", float(obs.vix)),
        Signal("raaal", subj, "avg_correlation", float(obs.avg_correlation)),
        Signal("raaal", subj, "breadth", float(obs.breadth)),
        Signal("raaal", subj, "drawdown", float(obs.drawdown)),
        Signal("raaal", subj, "expected_shortfall", float(obs.expected_shortfall)),
        Signal("raaal", subj, "cash_weight", float(obs.cash_weight)),
        Signal("raaal", subj, "gross_leverage", float(obs.gross_leverage)),
        Signal("raaal", subj, "portfolio_drift", float(obs.portfolio_drift)),
        Signal("raaal", subj, "hhi", float(obs.hhi)),
        Signal("raaal", subj, "max_weight", float(obs.max_weight)),
        Signal("raaal", subj, "rule_ml_agreement", float(obs.rule_ml_agreement)),
        Signal("raaal", subj, "net_rebalance_benefit", float(obs.expected_benefit - obs.rebalance_cost)),
    ]
    # data freshness (a stale feed is a discovery in its own right)
    sigs.append(Signal("raaal", subj, "data_age_days", float(obs.data_age_days)))
    # regime change delivered as a graph signal (GraphChangeDetector picks up value>0)
    if obs.prev_regime is not None and obs.prev_regime != obs.regime:
        sigs.append(Signal("raaal", subj, "graph.regime_changed", 1.0,
                           dimensions={"from": obs.prev_regime, "to": obs.regime}))
    # rule-vs-ML regime divergence
    if obs.rule_ml_agreement < 1.0:
        sigs.append(Signal("raaal", subj, "graph.regime_divergence", 1.0))
    # per-strategy trailing excess vs benchmark (StrategyDegradationDetector reads these)
    for sid, excess in obs.strategy_excess.items():
        sigs.append(Signal("raaal", f"strategy.{sid}", "excess_return", float(excess)))
    return sigs
