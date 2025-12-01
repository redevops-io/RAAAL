"""Portfolio optimization engine (Sharpe maximization with guardrails)."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_RF, REGIME_CONSTRAINTS, TICKER_INDEX, UNIVERSE
from .features import exponential_cov, exponential_mean


def _bounds(regime: str) -> Tuple[Tuple[float, float], ...]:
    guardrails = REGIME_CONSTRAINTS[regime]
    bnds = []
    for asset in UNIVERSE:
        low = asset.lower
        high = asset.upper
        if asset.is_inverse:
            high = min(high, guardrails["inverse_cap"])
        if asset.asset_class == "cash":
            low = max(low, guardrails["cash_min"])
            high = min(high, guardrails["cash_max"])
        bnds.append((low, high))
    return tuple(bnds)


def _inverse_sum_constraint(regime: str):
    guardrails = REGIME_CONSTRAINTS[regime]
    def func(weights: np.ndarray) -> float:
        total = 0.0
        for idx, asset in enumerate(UNIVERSE):
            if asset.is_inverse:
                total += weights[idx]
        return guardrails["inverse_cap"] - total
    return func


def optimize_weights(
    returns: pd.DataFrame,
    regime: str,
    rf_rate: float | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.1,
) -> Dict[str, float]:
    tickers = [asset.ticker for asset in UNIVERSE]
    returns = returns[tickers].dropna()
    mu = exponential_mean(returns)
    cov = exponential_cov(returns)
    rf = rf_rate if rf_rate is not None else DEFAULT_RF

    def objective(weights: np.ndarray) -> float:
        port_ret = float(np.dot(mu.values, weights)) - rf
        port_vol = float(np.sqrt(weights.T @ cov.values @ weights))
        if port_vol == 0:
            return 1e6
        sharpe = port_ret / port_vol
        penalty = 0.0
        if prev_weights is not None:
            penalty = turnover_penalty * np.sum(np.abs(weights - prev_weights))
        return -sharpe + penalty

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget
        {"type": "ineq", "fun": _inverse_sum_constraint(regime)},
    ]

    bounds = _bounds(regime)
    x0 = np.array([max(low, 1.0 / len(UNIVERSE)) for low, _ in bounds])
    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-6},
    )

    if not res.success:
        return _risk_parity_fallback(cov)

    weights = res.x
    return {ticker: float(weight) for ticker, weight in zip(tickers, weights)}


def _risk_parity_fallback(cov: pd.DataFrame) -> Dict[str, float]:
    diag = np.diag(cov.values).copy()
    diag[diag <= 0] = 1e-6
    inv_vol = 1.0 / np.sqrt(diag)
    weights = inv_vol / inv_vol.sum()
    allocations = {asset.ticker: float(weight) for asset, weight in zip(UNIVERSE, weights)}
    return allocations


def optimize_weights_unrestricted(mu: pd.Series, cov: pd.DataFrame, rf: float) -> Dict[str, float]:
    """Maximize Sharpe ratio without regime guardrails while staying long-only."""
    tickers = [asset.ticker for asset in UNIVERSE]
    mu_vec = mu.reindex(tickers).values
    cov_matrix = cov.reindex(index=tickers, columns=tickers).values

    def objective(weights: np.ndarray) -> float:
        port_ret = float(np.dot(mu_vec, weights)) - rf
        port_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
        if port_vol == 0:
            return 1e6
        sharpe = port_ret / port_vol
        return -sharpe

    n_assets = len(tickers)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets  # long-only, fully invested per user definition
    x0 = np.full(n_assets, 1.0 / n_assets)
    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-6},
    )

    if not res.success:
        return _risk_parity_fallback(cov)

    return {ticker: float(weight) for ticker, weight in zip(tickers, res.x)}
