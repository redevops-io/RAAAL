"""Composite FOMO vs. FOBI indicator built from market structure signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    FOMO_COMPONENT_WEIGHTS,
    FOMO_LONG_LOOKBACK,
    FOMO_SCORE_THRESHOLDS,
    FOMO_SHORT_LOOKBACK,
)

ScoreTuple = Tuple[pd.Series, pd.Series]


@dataclass(frozen=True)
class FomoFobiSnapshot:
    """Lightweight container for the latest indicator reading."""

    timestamp: pd.Timestamp
    score: float
    state: str
    components: Dict[str, float]


_REQUIRED_TICKERS = {
    "SPY",
    "RSP",
    "QQQ",
    "MGK",
    "IWM",
    "BIL",
    "BRK-B",
    "HYG",
    "LQD",
    "TIP",
}

_OPTIONAL_TICKERS = {"^VIX", "^VVIX"}


def _window_return(series: pd.Series, window: int) -> pd.Series:
    return series.div(series.shift(window)) - 1.0


def _relative_performance(prices: pd.DataFrame, numerator: str, denominator: str, window: int) -> pd.Series:
    if numerator not in prices.columns or denominator not in prices.columns:
        return pd.Series(np.nan, index=prices.index)
    return _window_return(prices[numerator], window) - _window_return(prices[denominator], window)


def _volatility_complacency(prices: pd.DataFrame) -> pd.Series:
    if "SPY" not in prices.columns or "^VIX" not in prices.columns:
        return pd.Series(np.nan, index=prices.index)
    spy_returns = np.log(prices["SPY"]).diff()
    realized = spy_returns.rolling(FOMO_SHORT_LOOKBACK).std() * np.sqrt(252) * 100.0
    vix = prices["^VIX"].reindex_like(realized)
    return realized - vix


def _options_hedging_pressure(prices: pd.DataFrame) -> pd.Series:
    if "^VIX" not in prices.columns or "^VVIX" not in prices.columns:
        return pd.Series(np.nan, index=prices.index)
    vix = prices["^VIX"].replace(0.0, np.nan)
    vvix = prices["^VVIX"].replace(0.0, np.nan)
    spread = (vix - vvix).rolling(FOMO_SHORT_LOOKBACK).mean()
    return spread


def _cross_sectional_dispersion(prices: pd.DataFrame) -> pd.Series:
    subset = prices[[col for col in ["SPY", "QQQ", "IWM", "RSP"] if col in prices.columns]].copy()
    if subset.empty:
        return pd.Series(np.nan, index=prices.index)
    returns = np.log(subset).diff()
    dispersion = returns.std(axis=1)
    smoothed = dispersion.rolling(FOMO_SHORT_LOOKBACK).mean()
    return -smoothed


def _rolling_zscore(series: pd.Series, window: int = FOMO_LONG_LOOKBACK) -> pd.Series:
    if series.empty:
        return series.copy()
    mean = series.rolling(window=window, min_periods=max(window // 6, 21)).mean()
    std = series.rolling(window=window, min_periods=max(window // 6, 21)).std()
    z = (series - mean) / std
    return z.clip(-3.0, 3.0)


def _logistic(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def required_tickers() -> List[str]:
    extra = sorted(_OPTIONAL_TICKERS)
    return sorted(_REQUIRED_TICKERS.union(extra))


def compute_component_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """Return raw + z-scored component series for the indicator."""

    missing = [ticker for ticker in _REQUIRED_TICKERS if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Missing required tickers for FOMO/FOBI indicator: {missing}")

    df = pd.DataFrame(index=prices.index)
    df["component_breadth"] = _relative_performance(prices, "SPY", "RSP", FOMO_SHORT_LOOKBACK)
    df["component_mega_cap"] = _relative_performance(prices, "MGK", "RSP", FOMO_SHORT_LOOKBACK)
    df["component_tech_leadership"] = _relative_performance(prices, "QQQ", "SPY", FOMO_SHORT_LOOKBACK)
    df["component_small_cap_leadership"] = _relative_performance(prices, "IWM", "SPY", FOMO_SHORT_LOOKBACK)
    df["component_dispersion"] = _cross_sectional_dispersion(prices)
    df["component_cash_shortage"] = _relative_performance(prices, "SPY", "BIL", FOMO_SHORT_LOOKBACK)
    df["component_liquidity_stress"] = _relative_performance(prices, "HYG", "LQD", FOMO_SHORT_LOOKBACK)
    df["component_berkshire_cash"] = _relative_performance(prices, "SPY", "BRK-B", FOMO_LONG_LOOKBACK)
    df["component_vol_complacency"] = _volatility_complacency(prices)
    df["component_options_hedging"] = _options_hedging_pressure(prices)

    for column in list(df.columns):
        df[f"{column}_z"] = _rolling_zscore(df[column])

    return df


def compute_fomo_fobi_indicator(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate component z-scores into a composite sentiment series."""

    components = compute_component_scores(prices)
    z_cols = [col for col in components.columns if col.endswith("_z")]
    composite = pd.Series(0.0, index=components.index, dtype=float)
    weight_sum = sum(FOMO_COMPONENT_WEIGHTS.values())
    if weight_sum <= 0:
        raise ValueError("FOMO component weights must sum to positive value")

    for name, weight in FOMO_COMPONENT_WEIGHTS.items():
        column = f"component_{name}_z"
        if column not in components.columns:
            continue
        composite = composite.add(components[column].fillna(0.0) * weight, fill_value=0.0)

    composite = composite / weight_sum
    indicator = components.copy()
    indicator["fomo_fobi_score_raw"] = composite
    indicator["fomo_fobi_score"] = composite.clip(-3.0, 3.0)
    indicator["fomo_probability"] = _logistic(indicator["fomo_fobi_score"]).clip(0.0, 1.0)

    hi = FOMO_SCORE_THRESHOLDS["fomo"]
    lo = FOMO_SCORE_THRESHOLDS["fobi"]
    indicator["fomo_fobi_state"] = np.select(
        [indicator["fomo_fobi_score"] >= hi, indicator["fomo_fobi_score"] <= lo],
        ["FOMO", "FOBI"],
        default="neutral",
    )
    indicator["fomo_flag"] = indicator["fomo_fobi_state"] == "FOMO"
    indicator["fobi_flag"] = indicator["fomo_fobi_state"] == "FOBI"
    return indicator


def latest_snapshot(prices: pd.DataFrame) -> FomoFobiSnapshot:
    indicator = compute_fomo_fobi_indicator(prices)
    latest = indicator.dropna(how="all").iloc[-1]
    components = {
        key.replace("component_", "").replace("_z", ""): float(value)
        for key, value in latest.items()
        if key.startswith("component_") and key.endswith("_z")
    }
    return FomoFobiSnapshot(
        timestamp=indicator.index[-1],
        score=float(latest.get("fomo_fobi_score", 0.0)),
        state=str(latest.get("fomo_fobi_state", "neutral")),
        components=components,
    )