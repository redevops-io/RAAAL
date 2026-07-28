"""P4: deterministic investment discovery turns market/portfolio changes into proposed missions."""
from __future__ import annotations

from src.agentic.discovery_runtime import InvestmentDiscovery
from src.agentic.signals import PortfolioObservation, build_signals


def test_signals_cover_the_monitored_metrics():
    obs = PortfolioObservation(regime="risk_off", prev_regime="risk_on",
                               strategy_excess={"dual_momentum": -0.03})
    metrics = {s.metric for s in build_signals(obs)}
    assert {"vix", "drawdown", "cash_weight", "portfolio_drift", "hhi",
            "graph.regime_changed", "net_rebalance_benefit"} <= metrics
    assert any(s.subject == "strategy.dual_momentum" for s in build_signals(obs))


def test_discovery_creates_proposals_from_changes():
    obs = PortfolioObservation(
        regime="risk_off", prev_regime="risk_on",
        vix=35.0, avg_correlation=0.3, drawdown=-0.20, expected_shortfall=-0.12,
        cash_weight=0.02, data_age_days=5.0, portfolio_drift=0.30, hhi=0.40, max_weight=0.6,
        expected_benefit=0.010, rebalance_cost=0.001,
        strategy_excess={"dual_momentum": -0.05, "risk_parity": 0.01},
    )
    out = InvestmentDiscovery().cycle(obs)
    classes = {q["opportunity_class"] for q in out["queue"]}
    # every family of change surfaced as a proposal
    assert {"regime_change", "risk_limit_breach", "allocation_drift", "concentration_risk",
            "strategy_degradation", "rebalance_opportunity", "data_freshness_failure"} <= classes
    assert all(q["decision"] == "propose" for q in out["queue"])
    # strategy degradation is attributed to the degrading strategy only
    degr = [q for q in out["queue"] if q["opportunity_class"] == "strategy_degradation"]
    assert degr and all(q["subject"] == "strategy.dual_momentum" for q in degr)
    # queue is value-ranked and carries evidence
    vals = [q["expected_value"] for q in out["queue"]]
    assert vals == sorted(vals, reverse=True)
    assert all(q["evidence"] for q in out["queue"])


def test_calm_market_proposes_little():
    calm = PortfolioObservation()  # defaults: calm, no regime change, no breaches
    out = InvestmentDiscovery().cycle(calm)
    classes = {q["opportunity_class"] for q in out["queue"]}
    # no risk-limit breach / regime change / degradation in a calm cycle
    assert "risk_limit_breach" not in classes
    assert "regime_change" not in classes
