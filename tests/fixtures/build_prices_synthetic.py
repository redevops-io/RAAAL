"""Build the committed synthetic price fixture.

Run when the fixture must change; the output is committed, not generated at test
time. Generating on the fly would make every run depend on this file's current
behaviour, and a fixture whose content moves with the code cannot detect a
regression in the code.

    python3 tests/fixtures/build_prices_synthetic.py

The data is **invented**. It is shaped like market data — NYSE sessions, a
plausible factor structure, an inception gap, a split-like discontinuity — so the
evaluation stack has something realistic to run on, and it is deliberately not
calibrated to any real security. Nothing measured on it is a claim about
anything, which is exactly why it is safe to redistribute.

Construction is a two-factor model with fixed seeds:

    market factor      one common driver, so correlations are non-trivial and
                       the covariance estimator has real structure to find
    idiosyncratic      per-asset noise with declared volatilities
    inverse pairs      SH is built as the negative of SPY's factor loading, so
                       the universe contains a genuinely diversifying asset
                       rather than nine variations of one series

Every parameter below is fixed. Same script, same bytes, same SHA-256 — asserted
by a test, because a fixture that silently changes turns every downstream
assertion into a moving target.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calendars import CalendarRegistry  # noqa: E402

OUT = Path(__file__).parent / "prices_synthetic.parquet"

START, END = "2016-01-01", "2025-11-20"
"""Matches the evaluation protocols. A shorter span would leave the 504-day
long-warmup protocol with no post-warmup window to measure."""

SEED = 20260801

#: symbol -> (start price, market beta, annual idiosyncratic vol, annual drift)
#: The nine-asset methodology universe first, then the wider set the library
#: pages reference. Betas are chosen so the universe spans a real range rather
#: than clustering: equity long, equity inverse, duration, credit, commodity.
ASSETS = {
    "SPY":  (200.0,  1.00, 0.10,  0.08),
    "SH":   ( 45.0, -1.00, 0.10, -0.09),
    "TLT":  (120.0, -0.30, 0.12,  0.01),
    "TBT":  ( 35.0,  0.30, 0.24, -0.03),
    "LQD":  (115.0, -0.05, 0.06,  0.02),
    "DBC":  ( 15.0,  0.35, 0.16,  0.01),
    "GLD":  (110.0,  0.05, 0.13,  0.05),
    "HYG":  ( 85.0,  0.45, 0.07,  0.03),
    "BIL":  ( 91.0,  0.00, 0.001, 0.015),
    "TIP":  (112.0, -0.10, 0.05,  0.02),
    "RSP":  ( 78.0,  0.98, 0.11,  0.07),
    "QQQ":  (105.0,  1.15, 0.14,  0.11),
    "MGK":  ( 80.0,  1.12, 0.14,  0.11),
    "IWM":  (110.0,  1.05, 0.15,  0.06),
    "BRK-B":(130.0,  0.85, 0.12,  0.08),
    "BTC-USD": (430.0, 0.60, 0.60, 0.25),
    "^VIX":  ( 20.0, -3.00, 0.80,  0.00),
    "^VVIX": ( 90.0, -1.50, 0.45,  0.00),
}

#: An asset that did not exist at the start of the window. Its leading NaNs are
#: the missing-value case: code that assumes a complete rectangle breaks here
#: rather than in production. Kept outside the methodology universe so it
#: exercises loaders and frames without changing any evaluated result.
LATE_INCEPTION = ("BRK-B", "2017-06-15")

#: A 4:1 split-like discontinuity — a price series that jumps without a return
#: to match. Also outside the methodology universe, for the same reason.
DISCONTINUITY = ("MGK", "2021-03-15", 0.25)

TRADING_DAYS = 252.0


def build() -> pd.DataFrame:
    sessions = CalendarRegistry().resolve("nyse@1").sessions(
        pd.date_range(START, END, freq="D"))
    n = len(sessions)
    rng = np.random.default_rng(SEED)

    # One common driver. Without it every asset is independent, the correlation
    # matrix is the identity, and hierarchical clustering has nothing to cluster.
    market = rng.normal(0.08 / TRADING_DAYS, 0.13 / np.sqrt(TRADING_DAYS), n)

    columns = {}
    for symbol, (price0, beta, vol, drift) in ASSETS.items():
        idio = rng.normal(0.0, vol / np.sqrt(TRADING_DAYS), n)
        returns = drift / TRADING_DAYS + beta * market + idio
        series = price0 * np.exp(np.cumsum(returns))
        columns[symbol] = series

    frame = pd.DataFrame(columns, index=sessions)

    symbol, date, factor = DISCONTINUITY
    frame.loc[frame.index >= pd.Timestamp(date), symbol] *= factor

    symbol, date = LATE_INCEPTION
    frame.loc[frame.index < pd.Timestamp(date), symbol] = np.nan

    # Rounded to cents. Sub-cent precision is noise no venue would report, and
    # rounding roughly halves the compressed size.
    return frame.round(2)


def main() -> None:
    frame = build()
    frame.to_parquet(OUT, compression="zstd", index=True)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"{OUT}  {frame.shape[0]} sessions x {frame.shape[1]} assets  "
          f"{OUT.stat().st_size / 1024:.0f} KiB")
    print(f"sha256  {digest}")


if __name__ == "__main__":
    main()
