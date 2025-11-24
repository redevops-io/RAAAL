"""Lightweight macro nowcast proxies derived from the existing price universe."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

NOWCAST_LOOKBACK = 63


def _window_return(series: pd.Series | None, window: int = NOWCAST_LOOKBACK) -> float:
    if series is None or series.empty or len(series) <= window:
        return float("nan")
    start = series.iloc[-window]
    end = series.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return float(end / start - 1.0)


def _scale(value: float, max_abs: float = 0.15) -> float:
    if not np.isfinite(value):
        return 0.0
    normalized = value / max_abs
    return float(np.tanh(normalized))


def compute_nowcasts(prices: pd.DataFrame) -> Dict[str, float]:
    """Return bounded growth/inflation/liquidity tilts using available tickers."""

    def series(name: str) -> pd.Series | None:
        if name not in prices.columns:
            return None
        return prices[name].dropna()

    spy = series("SPY")
    iwm = series("IWM")
    bil = series("BIL")
    dbc = series("DBC")
    tip = series("TIP")
    hyg = series("HYG")
    lqd = series("LQD")
    tlt = series("TLT")

    growth_proxy = np.nanmean([_window_return(spy), _window_return(iwm)]) - _window_return(bil)
    inflation_proxy = _window_return(dbc) - _window_return(tip)
    liquidity_proxy = _window_return(hyg) - _window_return(lqd)
    real_rate_proxy = _window_return(tlt) - _window_return(tip)

    nowcasts = {
        "growth": _scale(growth_proxy),
        "inflation": _scale(inflation_proxy),
        "liquidity": _scale(liquidity_proxy),
        "real_rates": _scale(real_rate_proxy),
    }
    return {key: float(value) for key, value in nowcasts.items()}
