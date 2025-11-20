"""Shared helpers for portfolio analytics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import DEFAULT_RF, UNIVERSE


def rf_from_sgov(prices: pd.DataFrame) -> float:
    """Estimate daily risk-free rate using the cash proxy (BIL)."""
    cash_ticker = "BIL"
    if cash_ticker not in prices.columns:
        return DEFAULT_RF
    ret = prices[cash_ticker].pct_change().dropna()
    if ret.empty:
        return DEFAULT_RF
    annualized = ret.tail(21).mean() * 252
    return float(max(annualized, 0.0) / 252)


def weights_array(weights: Dict[str, float]) -> np.ndarray:
    return np.array([weights.get(asset.ticker, 0.0) for asset in UNIVERSE])


def portfolio_metrics(weights: Dict[str, float], mu: pd.Series, cov: pd.DataFrame, rf: float) -> Dict[str, float]:
    vec = weights_array(weights)
    ret = float(np.dot(mu.values, vec)) * 252
    vol = float(np.sqrt(vec.T @ cov.values @ vec)) * np.sqrt(252)
    sharpe = (ret - rf * 252) / vol if vol else 0.0

    spy_idx = None
    for idx, asset in enumerate(UNIVERSE):
        if asset.ticker == "SPY":
            spy_idx = idx
            break
    beta = 0.0
    if spy_idx is not None and cov.iloc[spy_idx, spy_idx] > 0:
        cov_with_spy = float(vec @ cov.iloc[:, spy_idx])
        beta = cov_with_spy / float(cov.iloc[spy_idx, spy_idx])

    return {"exp_return": ret, "exp_vol": vol, "sharpe": sharpe, "beta_proxy": beta}


def build_rationales(
    weights: Dict[str, float],
    mu: pd.Series,
    regime: str,
    diagnostics: Dict[str, float],
) -> Dict[str, str]:
    rationales = {}
    for asset in UNIVERSE:
        w = weights.get(asset.ticker, 0.0)
        if w < 0.01:
            continue
        trend = mu.get(asset.ticker, 0.0) * 252
        if asset.asset_class == "cash":
            rationale = f"Hold {w:.0%} cash to respect {regime} floor"
        elif asset.is_inverse:
            rationale = f"Inverse exposure to damp beta; trend={trend:.2%}"
        elif asset.asset_class == "commodities":
            rationale = f"Inflation hedge (trend={trend:.2%})"
        else:
            rationale = f"Momentum {trend:.2%} with regime={regime}"
        rationales[asset.ticker] = rationale
    return rationales
