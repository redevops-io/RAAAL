"""Alpha158-inspired technical feature set for ML forecasters.

Produces a rich feature DataFrame from price and volume data, drawing on
Qlib's Alpha158 factor library.  Each function returns a ``pd.DataFrame``
aligned on the original DatetimeIndex so features can be joined trivially.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import FAST_LOOKBACK, MED_LOOKBACK, SLOW_LOOKBACK

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, min_periods=window).mean()
    rs = _safe_div(gain, loss)
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


def macd_features(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


def rsi_features(close: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """RSI over several look-back windows."""
    windows = windows or [14, 30]
    frames = {}
    for w in windows:
        frames[f"rsi_{w}"] = _rsi(close, w)
    return pd.DataFrame(frames, index=close.index)


def bollinger_features(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with %B and bandwidth."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = _safe_div(close - lower, upper - lower)
    bandwidth = _safe_div(upper - lower, sma)
    return pd.DataFrame(
        {"boll_upper": upper, "boll_lower": lower, "boll_pct_b": pct_b, "boll_bw": bandwidth},
        index=close.index,
    )


def adx_features(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) and +DI / -DI components."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=window, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_smooth = pd.Series(plus_dm, index=close.index).ewm(span=window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=close.index).ewm(span=window, adjust=False).mean()

    plus_di = 100.0 * _safe_div(plus_dm_smooth, atr)
    minus_di = 100.0 * _safe_div(minus_dm_smooth, atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.ewm(span=window, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di, "atr": atr},
        index=close.index,
    )


def rolling_kurtosis(returns: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Rolling excess kurtosis — fat-tail detector."""
    windows = windows or [FAST_LOOKBACK, MED_LOOKBACK]
    frames = {}
    for w in windows:
        frames[f"kurt_{w}"] = returns.rolling(w, min_periods=max(w // 2, 10)).kurt()
    return pd.DataFrame(frames, index=returns.index)


def rolling_autocorrelation(returns: pd.Series, lags: Optional[List[int]] = None, window: int = MED_LOOKBACK) -> pd.DataFrame:
    """Rolling lag-n autocorrelation — mean-reversion / trend signal."""
    lags = lags or [1, 5, 21]
    frames = {}
    for lag in lags:
        frames[f"autocorr_{lag}"] = returns.rolling(window, min_periods=max(window // 2, 10)).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False
        )
    return pd.DataFrame(frames, index=returns.index)


def return_features(close: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Simple and log returns over various horizons."""
    windows = windows or [1, 5, 10, 21, 63]
    frames = {}
    for w in windows:
        frames[f"ret_{w}d"] = close.pct_change(w)
        frames[f"logret_{w}d"] = np.log(close / close.shift(w))
    return pd.DataFrame(frames, index=close.index)


def volatility_features(returns: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Rolling realized volatility and vol-of-vol."""
    windows = windows or [FAST_LOOKBACK, MED_LOOKBACK]
    frames = {}
    for w in windows:
        vol = returns.rolling(w, min_periods=max(w // 2, 10)).std() * np.sqrt(252)
        frames[f"vol_{w}"] = vol
        frames[f"vol_of_vol_{w}"] = vol.rolling(w, min_periods=max(w // 2, 10)).std()
    return pd.DataFrame(frames, index=returns.index)


def moving_average_features(close: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """SMA cross-over signals and distance from moving averages."""
    windows = windows or [5, 10, 20, 50, 200]
    frames = {}
    for w in windows:
        sma = close.rolling(w).mean()
        frames[f"sma_{w}"] = sma
        frames[f"close_over_sma_{w}"] = _safe_div(close, sma)
    return pd.DataFrame(frames, index=close.index)


def volume_features(volume: pd.Series, close: pd.Series, windows: Optional[List[int]] = None) -> pd.DataFrame:
    """Volume-based features: OBV trend, VWAP-style, volume ratio."""
    windows = windows or [FAST_LOOKBACK, MED_LOOKBACK]
    frames: dict = {}
    if volume is None or volume.isna().all():
        return pd.DataFrame(index=close.index)
    for w in windows:
        frames[f"vol_ratio_{w}"] = _safe_div(volume, volume.rolling(w).mean())
    obv = (np.sign(close.diff()) * volume).cumsum()
    frames["obv"] = obv
    frames["obv_ema"] = _ema(obv, 21)
    return pd.DataFrame(frames, index=close.index)


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------


def build_alpha_features(
    prices: pd.DataFrame,
    ticker: str,
    *,
    include_volume: bool = False,
) -> pd.DataFrame:
    """Build a full Alpha158-style feature set for a single ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Multi-column price DataFrame (must have at least ``ticker`` column).
    ticker : str
        The ticker symbol to build features for.
    include_volume : bool
        Whether to include volume-derived features (requires 'Volume' column).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by date, columns prefixed with ``ticker_``.
    """
    if ticker not in prices.columns:
        return pd.DataFrame(index=prices.index)

    close = prices[ticker].dropna()
    if close.empty:
        return pd.DataFrame(index=prices.index)

    log_returns = np.log(close / close.shift(1))

    parts = [
        macd_features(close),
        rsi_features(close),
        bollinger_features(close),
        return_features(close),
        volatility_features(log_returns),
        rolling_kurtosis(log_returns),
        rolling_autocorrelation(log_returns),
        moving_average_features(close),
    ]

    # ADX requires high/low — approximate from close if unavailable
    high = prices.get(f"{ticker}_High", close * 1.005)
    low = prices.get(f"{ticker}_Low", close * 0.995)
    parts.append(adx_features(high, low, close))

    if include_volume:
        vol_col = prices.get(f"{ticker}_Volume")
        if vol_col is not None:
            parts.append(volume_features(vol_col, close))

    combined = pd.concat(parts, axis=1)
    combined.columns = [f"{ticker}_{col}" for col in combined.columns]
    return combined.reindex(prices.index)


def build_universe_features(
    prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    include_volume: bool = False,
) -> pd.DataFrame:
    """Build Alpha158 features for every ticker in the universe.

    Returns a single wide DataFrame suitable for feeding into ML forecasters.
    """
    from .config import ordered_tickers  # avoid circular import at module level

    tickers = tickers or ordered_tickers()
    frames = [build_alpha_features(prices, t, include_volume=include_volume) for t in tickers]
    return pd.concat(frames, axis=1)
