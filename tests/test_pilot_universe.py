"""The committed tradable universe (data/instruments/pilot_universe.yaml) — the
source of record for what the snapshot prices, and therefore what a plan may
name. Committed so the daily refresh has a universe in a fresh checkout, and so
adding an instrument (the top-100 S&P 500 equities) is a reviewed edit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
UNIVERSE = REPO / "data" / "instruments" / "pilot_universe.yaml"

_SPEC = importlib.util.spec_from_file_location(
    "build_catalog_snapshot", REPO / "scripts" / "build_catalog_snapshot.py")
bcs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bcs
_SPEC.loader.exec_module(bcs)  # type: ignore[union-attr]

# The universe that was live in the S3 snapshot before the equities were added —
# none of it may be dropped, or a plan that used to price stops.
LIVE_BEFORE = {
    "AAPL","AGG","AMZN","AOR","BIL","BND","BNDW","BRK-B","BTC-USD","DBC","DIA",
    "ESGU","GLD","GOOGL","HYG","IEF","IEI","IWB","IWM","LQD","META","MGK","MTUM",
    "MUB","NVDA","ORCL","QQQ","QUAL","RSP","SCHB","SGOV","SH","SHY","SPY","TBT",
    "TIP","TLT","USMV","VBR","VEA","VIG","VNQ","VOO","VT","VTI","VTV","VWO","VXUS",
}


def test_the_universe_file_has_funds_and_equities():
    doc = yaml.safe_load(UNIVERSE.read_text())
    assert doc["funds_and_macro"], "no funds/macro"
    assert len(doc["equities"]) >= 100, "the top-100 equities are the point"


def test_it_prices_the_requested_single_stocks():
    tickers = set(bcs.universe_tickers())
    # The report that started this: MSFT, and the mega-caps beside it.
    for want in ("MSFT", "AAPL", "NVDA", "TSLA", "JPM", "V", "UNH", "WMT"):
        assert want in tickers, f"{want} is not in the priced universe"


def test_no_previously_live_ticker_is_dropped():
    tickers = set(bcs.universe_tickers())
    dropped = LIVE_BEFORE - tickers
    assert not dropped, f"these were priced and would stop: {sorted(dropped)}"


def test_the_universe_is_deduped_and_reasonable():
    tickers = bcs.universe_tickers()
    assert len(tickers) == len(set(tickers)), "duplicate tickers"
    assert 130 <= len(tickers) <= 200, f"unexpected size {len(tickers)}"
