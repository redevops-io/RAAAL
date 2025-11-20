"""Feature engineering helpers for the allocation pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAST_LOOKBACK, MED_LOOKBACK, PRICE_COLUMN, SLOW_LOOKBACK


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def annualized_vol(returns: pd.Series, window: int = MED_LOOKBACK) -> float:
    rolling = returns.rolling(window=window).std().dropna()
    if rolling.empty:
        return float("nan")
    return float(rolling.iloc[-1] * np.sqrt(252))


def rolling_sharpe(
    returns: pd.Series,
    rf: float,
    window: int = MED_LOOKBACK,
) -> float:
    if returns.shape[0] < window:
        return float("nan")
    windowed = returns.iloc[-window:]
    excess = windowed - rf
    mean = excess.mean() * 252
    vol = excess.std() * np.sqrt(252)
    if vol == 0:
        return 0.0
    return float(mean / vol)


def beta_to_market(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = SLOW_LOOKBACK,
) -> float:
    if len(asset_returns) < window or len(market_returns) < window:
        return float("nan")
    ar = asset_returns.iloc[-window:]
    mr = market_returns.iloc[-window:]
    cov = np.cov(ar, mr)
    var = np.var(mr)
    if var == 0:
        return 0.0
    return float(cov[0, 1] / var)


def exponential_mean(returns: pd.DataFrame, span: int = MED_LOOKBACK) -> pd.Series:
    return returns.ewm(span=span, adjust=False).mean().iloc[-1]


def exponential_cov(returns: pd.DataFrame, span: int = MED_LOOKBACK) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
    idx = returns.index
    decay = np.exp(-np.arange(len(idx))[::-1] / span)
    weights = decay / decay.sum()
    centered = returns - returns.mean()
    weighted = centered.mul(weights, axis=0)
    cov = centered.T.dot(weighted)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def corr_spy_tlt(returns: pd.DataFrame, window: int = MED_LOOKBACK) -> float:
    if {"SPY", "TLT"}.issubset(returns.columns):
        sub = returns[["SPY", "TLT"]].dropna()
        if sub.shape[0] >= window:
            return float(sub.iloc[-window:].corr().iloc[0, 1])
    return float("nan")


def credit_spread_momentum(returns: pd.DataFrame, window: int = FAST_LOOKBACK) -> float:
    if {"LQD", "TLT"}.issubset(returns.columns):
        spread = returns["LQD"] - returns["TLT"]
        roll = spread.rolling(window=window).mean().dropna()
        if not roll.empty:
            return float(roll.iloc[-1])
    return 0.0


def commodity_momentum(returns: pd.DataFrame, window: int = MED_LOOKBACK) -> float:
    if "DBC" in returns.columns:
        roll = returns["DBC"].rolling(window=window).mean().dropna()
        if not roll.empty:
            return float(roll.iloc[-1])
    return 0.0


def tip_momentum(prices: pd.DataFrame, window: int = MED_LOOKBACK) -> float:
    if "TIP" not in prices.columns:
        return 0.0
    pct = prices["TIP"].pct_change().rolling(window=window).mean().dropna()
    if pct.empty:
        return 0.0
    return float(pct.iloc[-1])
