"""Download and cache ETF price history."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

from .config import PRICE_COLUMN

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('^', 'IDX')}.parquet"


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.set_index("Date")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    path = _cache_path(ticker)
    df_to_store = df.copy()
    df_to_store.reset_index().to_parquet(path, index=False)


def download_prices(
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return panel of adjusted close prices for all tickers between start/end."""

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        cache = None if force_refresh else _load_cache(ticker)
        needs_download = True
        if cache is not None and not cache.empty:
            last_ts = cache.index.max()
            if last_ts >= end - timedelta(days=2):
                needs_download = False
        if needs_download:
            hist = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="column",
                progress=False,
            )
            if hist.empty:
                raise RuntimeError(f"No data downloaded for {ticker}")
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist.columns = [col[0] if isinstance(col, tuple) else col for col in hist.columns]
            _save_cache(ticker, hist)
            cache = hist
        frames.append(cache[[PRICE_COLUMN]].rename(columns={PRICE_COLUMN: ticker}))

    prices = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    prices = prices.ffill().dropna()
    return prices
