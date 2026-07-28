"""Investment Discovery Runtime: wire the detectors into agentic_os DiscoveryRuntime and shape output.

One cycle: observe -> detect -> propose. Each actionable proposal is relabelled to a portfolio
opportunity_class (regime_change / risk_limit_breach / allocation_drift / strategy_degradation /
concentration_risk / data_freshness_failure / rebalance_opportunity) and emitted in the console queue
contract so the operating console renders it in the Attention Queue.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from agentic_os.discovery import DiscoveryRuntime
from agentic_os.discovery.policies import ProposalPolicy

from .detectors import investment_detectors
from .signals import PortfolioObservation, build_signals

# detection metric/kind -> portfolio opportunity class (the discovery vocabulary)
_CLASS_BY_METRIC = {
    "graph.regime_changed": "regime_change",
    "graph.regime_divergence": "regime_change",
    "drawdown": "risk_limit_breach",
    "expected_shortfall": "risk_limit_breach",
    "cash_weight": "risk_limit_breach",
    "gross_leverage": "risk_limit_breach",
    "vix": "risk_limit_breach",
    "avg_correlation": "risk_limit_breach",
    "portfolio_drift": "allocation_drift",
    "hhi": "concentration_risk",
    "max_weight": "concentration_risk",
    "data_age_days": "data_freshness_failure",
    "net_rebalance_benefit": "rebalance_opportunity",
    "excess_return": "strategy_degradation",
}
_BASE_VALUE = {
    "regime_change": 6000.0, "risk_limit_breach": 7000.0, "allocation_drift": 2500.0,
    "concentration_risk": 4000.0, "strategy_degradation": 3500.0, "rebalance_opportunity": 2000.0,
    "data_freshness_failure": 1500.0,
}


class InvestmentDiscovery:
    """Deterministic portfolio discovery over one observation, mirroring revenue-agent's loop."""

    def __init__(self, grants: Optional[List[str]] = None) -> None:
        self._grants = grants or ["read:market", "read:portfolio"]

    def _runtime(self) -> DiscoveryRuntime:
        # a fresh store each cycle so detections reflect the current observation window
        return DiscoveryRuntime(detectors=investment_detectors(),
                                policy=ProposalPolicy(), grants=self._grants)

    def cycle(self, obs: PortfolioObservation) -> Dict[str, object]:
        rt = self._runtime()
        result = rt.cycle(build_signals(obs), handoff=False)
        # Build the queue straight from the per-metric DETECTIONS (precise metric -> class mapping),
        # grouped by (opportunity_class, subject). The proposal layer still runs for suppression/scoring;
        # its counts are reported in the summary.
        groups: Dict[tuple, Dict[str, object]] = {}
        for det in result.detections:
            klass = _CLASS_BY_METRIC.get(det.metric)
            if klass is None:
                continue
            key = (klass, det.subject)
            g = groups.setdefault(key, {"sev": 0.0, "evidence": [], "metrics": []})
            g["sev"] = max(float(g["sev"]), float(det.severity))          # type: ignore[arg-type]
            g["evidence"].extend(det.evidence or [])                       # type: ignore[union-attr]
            g["metrics"].append(det.metric)                                # type: ignore[union-attr]

        queue: List[Dict[str, object]] = []
        for (klass, subject), g in groups.items():
            sev = float(g["sev"])                                          # type: ignore[arg-type]
            base = _BASE_VALUE.get(klass, 3000.0)
            queue.append({
                "subject": subject,
                "opportunity_class": klass,
                "score": round(sev, 4),
                "confidence": round(min(0.99, 0.6 + 0.4 * sev), 3),
                "urgency": round(sev, 3),
                "risk": round(sev, 3),
                "expected_value": round(base * sev, 1),
                "decision": "propose",
                "regime": obs.regime,
                "metrics": sorted(set(g["metrics"])),                      # type: ignore[arg-type]
                "evidence": (g["evidence"] or [])[:4],                     # type: ignore[index]
            })
        queue.sort(key=lambda q: q["expected_value"], reverse=True)
        return {
            "queue": queue,
            "detections": len(result.detections),
            "proposals": len(result.proposals),
            "actionable": len(queue),
            "regime": obs.regime,
        }
