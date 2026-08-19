"""Turn a custom index's components into the level the engine prices against.

The valuation half of custom indices (the market-data half — which parts, and
their series — is `market_data.custom_indices`). Given a frame that already
carries the component columns, this computes the index as a fixed-weight
total-return combination of them, scaled to a base: the level a fund tracking
the index would report.

Fixed-weight and daily-combined, not buy-and-hold: an index is a rule, not a
portfolio that drifts. Each day's index return is the weighted sum of the
components' returns at the stated weights, and the level is the running product
of those returns from a base of 100. The base is arbitrary because the engine
buys a notional amount of the index — the share count scales with it and the
value tracks the index either way — so it is chosen for readability, not meaning.

Computed from whatever series the frame holds: on the reinvested (total-return)
frame this yields a total-return index, on the price frame a price index, which
is the correct pairing without this module having to know which it was handed.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..market_data.custom_indices import composition, is_custom_index

#: The level the index starts at. Arbitrary — see the module docstring.
INDEX_BASE = 100.0


def _normalised(parts):
    total = float(sum(parts.values())) or 1.0
    return {component: float(weight) / total
            for component, weight in parts.items()}


def materialise(prices: Any, symbols: Iterable[str]) -> Any:
    """`prices` with a computed column added for each custom index in `symbols`
    whose components are all present.

    Ordinary tickers are untouched. A custom index missing a component is left
    absent rather than approximated, so the engine reads it as no price history
    — the honest outcome, and the signal that the snapshot (or DeerFlow's
    composition) still owes a series. The frame is copied only if a column is
    actually added, so the common no-index path allocates nothing.
    """
    out = prices
    for symbol in dict.fromkeys(symbols):          # de-dupe, keep order
        if not is_custom_index(symbol) or symbol in out.columns:
            continue
        parts = composition(symbol)
        if not parts or not all(component in out.columns for component in parts):
            continue
        weights = _normalised(parts)
        daily_return = None
        for component, weight in weights.items():
            leg = out[component].pct_change().fillna(0.0) * weight
            daily_return = leg if daily_return is None else daily_return + leg
        level = INDEX_BASE * (1.0 + daily_return).cumprod()
        if out is prices:
            out = prices.copy()
        out[symbol] = level
    return out
