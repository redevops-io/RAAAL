"""What a custom basket may actually hold — the priced intersection.

`data/instruments/pilot_universe.yaml` declares what a plan *may name*; the vendor
snapshot is what is actually *priced*. Offering a security the snapshot does not
price is the silent-no-figure defect the whole evaluator is built to avoid: a
basket naming an unpriced ticker reads fine and then produces nothing. So the
basket composer offers only the intersection — the securities this snapshot prices
— and expanding the choice is a data job (add the ticker to the universe and
refresh the snapshot), not a UI one.

Read once from the committed reference snapshot's schema (columns only, no data),
so a render never pays for a price load and the list cannot drift from what the
engine can evaluate.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SNAPSHOT = Path("data/snapshots/prices-yahoo-20260822.parquet")
_UNIVERSE_YAML = Path("data/instruments/pilot_universe.yaml")

#: Friendly names for the securities a basket can hold. Tickers the map does not
#: cover fall back to the ticker itself — a label is a convenience, not identity.
_NAMES = {
    # individual equities the snapshot prices
    "AAPL": "Apple", "AMZN": "Amazon", "BRK-B": "Berkshire Hathaway",
    "GOOGL": "Alphabet", "META": "Meta", "NVDA": "NVIDIA", "ORCL": "Oracle",
    # the broad funds people recognise
    "SPY": "S&P 500", "VOO": "S&P 500 (Vanguard)", "VTI": "US Total Market",
    "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000", "RSP": "S&P 500 Equal Weight",
    "VEA": "Developed ex-US", "VWO": "Emerging Markets", "VXUS": "Total International",
    "VT": "Total World", "IWB": "Russell 1000", "SCHB": "US Broad Market",
    "MTUM": "Momentum Factor", "QUAL": "Quality Factor", "USMV": "Min Volatility",
    "VTV": "US Value", "VBR": "US Small-Cap Value", "VIG": "Dividend Growth", "ESGU": "US ESG",
    "AGG": "US Aggregate Bonds", "BND": "US Total Bond", "BNDW": "World Bond",
    "TLT": "Long Treasuries", "IEF": "7-10yr Treasuries", "IEI": "3-7yr Treasuries",
    "SHY": "1-3yr Treasuries", "SGOV": "0-3mo Treasuries", "BIL": "T-Bills",
    "LQD": "Investment-Grade Credit", "HYG": "High-Yield Credit", "MUB": "Municipal Bonds",
    "TIP": "TIPS", "AOR": "Growth Allocation (60/40)", "GLD": "Gold", "DBC": "Broad Commodities",
    "VNQ": "US Real Estate", "MGK": "Mega-Cap Growth", "BTC-USD": "Bitcoin",
    "SH": "Inverse S&P 500", "TBT": "Inverse Long Treasuries",
}


@lru_cache(maxsize=1)
def _priced() -> frozenset:
    """The tickers the reference snapshot actually prices (columns only)."""
    try:
        import pyarrow.parquet as pq
        return frozenset(pq.ParquetFile(_SNAPSHOT).schema.names)
    except Exception:
        try:
            import pandas as pd
            return frozenset(pd.read_parquet(_SNAPSHOT).columns)
        except Exception:
            return frozenset()


@lru_cache(maxsize=1)
def evaluable() -> dict:
    """`{equities: [...], funds: [...]}` — declared, priced, and named. Each item
    is `{ticker, name, kind}`; only securities the snapshot prices are included."""
    import yaml

    priced = _priced()
    try:
        declared = yaml.safe_load(_UNIVERSE_YAML.read_text()) or {}
    except Exception:
        declared = {}

    def rows(tickers, kind):
        return [
            {"ticker": t, "name": _NAMES.get(t, t), "kind": kind}
            for t in (tickers or []) if t in priced
        ]

    return {
        "equities": rows(declared.get("equities"), "stock"),
        "funds": rows(declared.get("funds_and_macro"), "fund"),
    }
