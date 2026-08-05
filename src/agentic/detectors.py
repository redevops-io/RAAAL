"""Investment discovery detectors (docx §11) - deterministic, no LLM.

Reuses the generic agentic_os detectors (Baseline / RateOfChange / Threshold / Freshness / GraphChange)
and adds three portfolio-specific ones that the generic set cannot express:
  * ConcentrationDetector       - systemic-centrality / HHI increase (concentration risk).
  * StrategyDegradationDetector - a registered strategy's trailing excess vs benchmark going negative.
  * RebalanceCostBenefitDetector- a rebalance whose expected benefit exceeds its cost.

LLMs are used only AFTER deterministic discovery (Sidekick explain), never to detect numerical anomalies.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from agentic_os_enterprise.discovery import Signal  # noqa: F401 (re-export convenience)
from agentic_os_enterprise.discovery.detectors import (
    BaselineDetector,
    Detection,
    Detector,
    GraphChangeDetector,
    RateOfChangeDetector,
    ThresholdDetector,
)
from agentic_os_enterprise.discovery import baselines as B
from agentic_os_enterprise.discovery.signals import SignalStore

from ..config import MANDATE_CONSTRAINTS


def _sev(x: float) -> float:
    import math
    return max(0.0, min(1.0, 1.0 - math.exp(-abs(x))))


class ConcentrationDetector:
    """Flags rising portfolio concentration (HHI / max single weight above a cap)."""

    name = "concentration"

    def __init__(self, *, hhi_cap: float = 0.28, max_weight_cap: float = 0.5) -> None:
        self._hhi_cap = hhi_cap
        self._maxw_cap = max_weight_cap

    def scan(self, store: SignalStore, only_subjects: Optional[set] = None) -> List[Detection]:
        out: List[Detection] = []
        for subject, metric in store.keys():
            if only_subjects is not None and subject not in only_subjects:
                continue
            if metric not in ("hhi", "max_weight"):
                continue
            latest = store.latest(subject, metric)
            if latest is None:
                continue
            cap = self._hhi_cap if metric == "hhi" else self._maxw_cap
            if latest.value <= cap:
                continue
            out.append(Detection(kind="concentration", subject=subject, metric=metric,
                                 severity=round(_sev((latest.value - cap) / max(cap, 1e-9) + 1.0), 3),
                                 expected=cap, observed=latest.value, detector=self.name,
                                 evidence=[{"cap": cap, "observed": latest.value,
                                            "note": "systemic concentration above mandate comfort"}]))
        return out


class StrategyDegradationDetector:
    """Flags a registered strategy whose trailing excess vs benchmark has turned negative."""

    name = "strategy_degradation"

    def __init__(self, *, floor: float = 0.0) -> None:
        self._floor = floor

    def scan(self, store: SignalStore, only_subjects: Optional[set] = None) -> List[Detection]:
        out: List[Detection] = []
        for subject, metric in store.keys():
            if metric != "excess_return" or not subject.startswith("strategy."):
                continue
            if only_subjects is not None and subject not in only_subjects:
                continue
            latest = store.latest(subject, metric)
            if latest is None or latest.value >= self._floor:
                continue
            out.append(Detection(kind="strategy_degradation", subject=subject, metric=metric,
                                 severity=round(_sev(abs(latest.value) * 4 + 0.5), 3),
                                 expected=self._floor, observed=latest.value, detector=self.name,
                                 evidence=[{"strategy": subject.split(".", 1)[-1],
                                            "trailing_excess": latest.value}]))
        return out


class RebalanceCostBenefitDetector:
    """Flags a rebalance OPPORTUNITY when expected benefit exceeds cost by a margin."""

    name = "rebalance_cost_benefit"

    def __init__(self, *, margin: float = 0.001) -> None:
        self._margin = margin

    def scan(self, store: SignalStore, only_subjects: Optional[set] = None) -> List[Detection]:
        out: List[Detection] = []
        for subject, metric in store.keys():
            if metric != "net_rebalance_benefit":
                continue
            if only_subjects is not None and subject not in only_subjects:
                continue
            latest = store.latest(subject, metric)
            if latest is None or latest.value <= self._margin:
                continue
            out.append(Detection(kind="rebalance_opportunity", subject=subject, metric=metric,
                                 severity=round(_sev(latest.value * 50 + 0.3), 3),
                                 expected=self._margin, observed=latest.value, detector=self.name,
                                 evidence=[{"net_benefit": latest.value,
                                            "note": "expected benefit exceeds expected rebalance cost"}]))
        return out


def _risk_thresholds() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    m = MANDATE_CONSTRAINTS
    return {
        "drawdown": (-0.15, None),                         # breach if drawdown < -15%
        "expected_shortfall": (-0.08, None),               # breach if daily CVaR worse than -8%
        "cash_weight": (float(m.get("minimum_cash", 0.05)), None),   # breach if below the cash floor
        "gross_leverage": (None, float(m.get("leverage_cap", 1.0))),  # breach above the leverage cap
        "vix": (None, 30.0),                               # stress marker
        "avg_correlation": (None, 0.85),                   # correlation shock (diversification failure)
        "data_age_days": (None, 3.0),                      # stale-feed / missing-series failure
        "portfolio_drift": (None, float(m.get("maximum_turnover", 0.25))),  # drift past the turnover cap
    }


def investment_detectors() -> List[Detector]:
    """The full deterministic investment detector set (generic reuse + 3 custom)."""
    return [
        BaselineDetector("market_anomaly", B.ewma_forecast, kind="anomaly",
                         metrics={"vix", "avg_correlation", "breadth", "portfolio_drift"}),
        RateOfChangeDetector(window=3, max_rel_change=0.4,
                             metrics={"vix", "avg_correlation", "portfolio_drift", "hhi"}),
        GraphChangeDetector(prefix="graph."),               # regime change / rule-vs-ML divergence
        ThresholdDetector(_risk_thresholds(), name="risk_limit"),
        ConcentrationDetector(),
        StrategyDegradationDetector(),
        RebalanceCostBenefitDetector(),
    ]
