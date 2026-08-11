"""P1: the objectives are SELECTION policies over the approved registry (Amendments §2/§4).

Asserts the three objectives use genuinely different scoring, every recommendation is
traceable to a REGISTERED strategy implementation (no free-form weights), and the hard
mandate holds on every allocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agentic.selection import (
    POLICIES,
    build_snapshot,
    select_all_objectives,
    select_for_objective,
)
from src.config import AUX_SERIES, MANDATE_CONSTRAINTS, OBJECTIVES, UNIVERSE
from src.features import compute_returns, exponential_cov, exponential_mean
from src.strategies import CAPABILITY_BY_ID, TICKERS, registry_for_objective

BASE = [a.ticker for a in UNIVERSE]
DEFENSIVE_FAMILIES = {"risk_based", "sentiment"}
RETURN_SEEKING_FAMILIES = {"momentum", "factor_based", "regime_mean_reversion", "ensemble", "optimizer"}


def _synthetic_prices(seed: int = 7, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    cols = list(dict.fromkeys(BASE + AUX_SERIES + ["GLD", "SPY", "BIL"]))
    data = {}
    # per-asset drift/vol so the ranking is non-degenerate
    drift = {t: rng.uniform(-0.0002, 0.0006) for t in cols}
    vol = {t: rng.uniform(0.004, 0.02) for t in cols}
    for t in cols:
        if t == "^VIX":
            data[t] = 14 + np.abs(rng.normal(0, 3, n)).cumsum() * 0 + rng.uniform(12, 22, n)
            continue
        if t == "BIL":
            data[t] = 100 * (1 + 0.00008) ** np.arange(n)  # near-flat cash proxy
            continue
        shocks = rng.normal(drift[t], vol[t], n)
        data[t] = 100 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(data, index=dates)


def _snapshot(regime: str = "risk_on"):
    prices = _synthetic_prices()
    returns = compute_returns(prices)
    base_returns = returns[BASE].dropna()
    mu = exponential_mean(base_returns)
    cov = exponential_cov(base_returns)
    return build_snapshot(prices, base_returns, regime, mu, cov, rf=0.0)


def test_policies_use_distinct_scores():
    # the three scoring functions rank the same stats differently
    lowrisk = {"ann_return": 0.03, "ann_vol": 0.04, "sharpe": 0.75, "sortino": 1.0,
               "max_drawdown": -0.03, "cvar95": -0.004}
    highret = {"ann_return": 0.22, "ann_vol": 0.25, "sharpe": 0.88, "sortino": 1.1,
               "max_drawdown": -0.35, "cvar95": -0.03}
    assert POLICIES["min_risk"].score(lowrisk) > POLICIES["min_risk"].score(highret)
    assert POLICIES["max_total_return"].score(highret) > POLICIES["max_total_return"].score(lowrisk)
    # return-to-risk keys on Sharpe/Sortino, not raw return
    assert POLICIES["max_return_to_risk"].score(highret) > POLICIES["max_return_to_risk"].score(
        {**highret, "sharpe": 0.1, "sortino": 0.1})


def test_every_objective_selects_a_registered_strategy():
    snap = _snapshot()
    plans = select_all_objectives(snap)
    assert set(plans) == set(OBJECTIVES)
    for obj, res in plans.items():
        # traceability: the selected id is a real registered capability (or an honest abstain)
        assert res.selected_strategy_id in CAPABILITY_BY_ID or res.selected_strategy_id == "no_action"
        assert res.evidence_snapshot_id == snap.snapshot_id
        # the score_table only ever contains registered ids
        for row in res.score_table:
            assert row["id"] in CAPABILITY_BY_ID


def test_selection_is_objective_appropriate():
    snap = _snapshot()
    plans = select_all_objectives(snap)
    # min_risk must pick a defensive strategy; max_total_return a return-seeking one
    min_cap = CAPABILITY_BY_ID[plans["min_risk"].selected_strategy_id]
    ret_cap = CAPABILITY_BY_ID[plans["max_total_return"].selected_strategy_id]
    assert min_cap.family in DEFENSIVE_FAMILIES, f"min_risk chose {min_cap.id} ({min_cap.family})"
    assert ret_cap.family in RETURN_SEEKING_FAMILIES, f"max_total_return chose {ret_cap.id} ({ret_cap.family})"
    # the eligible sets genuinely differ per objective
    min_ids = {c.id for c in registry_for_objective("min_risk", "risk_on")}
    ret_ids = {c.id for c in registry_for_objective("max_total_return", "risk_on")}
    assert min_ids != ret_ids and min_ids and ret_ids


def test_no_freeform_weights_and_mandate_holds():
    snap = _snapshot()
    for obj in OBJECTIVES:
        res = select_for_objective(obj, snap, allow_abstain=False)
        w = res.weights
        # weights only over the known universe, long-only, fully invested
        assert set(w) == set(TICKERS)
        assert all(v >= -1e-9 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # hard mandate: inverse cap, crypto cap, cash floor
        inv = sum(w[a.ticker] for a in UNIVERSE if a.is_inverse)
        assert inv <= MANDATE_CONSTRAINTS["inverse_exposure_cap"] + 1e-6
        crypto = sum(w[a.ticker] for a in UNIVERSE if a.asset_class == "crypto")
        assert crypto <= MANDATE_CONSTRAINTS["crypto_cap"] + 1e-6
        assert w["BIL"] >= MANDATE_CONSTRAINTS["minimum_cash"] - 1e-6


def test_approved_only_in_live_path():
    # DRL (shadow) must never appear in the approved-only eligible set
    for obj in OBJECTIVES:
        ids = {c.id for c in registry_for_objective(obj, "risk_on", promotion=("approved",))}
        assert "drl_portfolio" not in ids and "adaptive_rotation" not in ids
