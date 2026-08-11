"""P9: the selector benchmark - walk-forward, costs, regime breakdown, non-inferiority gate."""
from __future__ import annotations

from src.agentic.bench import OBJECTIVES, run_investment_bench
from src.strategies import CAPABILITY_BY_ID


def _run():
    return run_investment_bench(seed=7, warmup=300, step=80, horizon=20)


def test_bench_is_deterministic():
    a = run_investment_bench(seed=7, warmup=300, step=80, horizon=20).to_dict()
    b = run_investment_bench(seed=7, warmup=300, step=80, horizon=20).to_dict()
    assert a == b  # no wall-clock / RNG in the scoring path


def test_bench_reports_selector_vs_best_fixed_with_gate():
    res = _run().to_dict()
    assert res["rounds"] >= 3
    for obj in OBJECTIVES:
        po = res["per_objective"][obj]
        # strategy quality (best fixed, a registered strategy) is separated from selector quality
        assert po["best_fixed_strategy"] in CAPABILITY_BY_ID
        assert "selector_score" in po and "best_fixed_score" in po
        assert isinstance(po["non_inferior"], bool)     # the promotion gate
        assert "avg_turnover" in po                      # transaction-cost aware


def test_bench_constraint_compliance_is_total():
    res = _run().to_dict()
    # every selected allocation across the whole walk-forward satisfies the hard mandate
    assert res["constraint_compliance"] == 1.0


def test_bench_has_baselines_and_regime_breakdown():
    res = _run().to_dict()
    assert {"equal_weight", "sixty_forty", "all_cash", "current_raaal_sharpe"} <= set(res["baselines"])
    # regime-specific results, not one annualized number
    assert set(res["by_regime"]) & {"risk_on", "risk_off", "inflation"}
    for regime, per in res["by_regime"].items():
        assert set(per) == set(OBJECTIVES)
