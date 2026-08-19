"""Indices composed from instruments the snapshot already carries.

A custom index is not a line of its own in the data — it is a rule for combining
other lines. VT (total world) is US total market plus total international; there
is no separate VT series to fetch, and inventing one would be exactly the
fabrication the rest of market_data refuses. So it is computed from its parts.

The responsibility is split deliberately, and the split is the whole point:

  * **the data service owns composition and components** — which instruments an
    index is made of, in what proportion, and delivering their series. That is a
    market-data question: what is held, and what did it cost.
  * **the valuation engine owns the calculation** — turning those component
    series into an index level. That is a valuation question, and it lives in
    `evaluation.index_calc`, not here.

This module is the first half. It resolves a symbol to its parts and refuses to
name one whose parts the snapshot does not carry, because an index the data
cannot compute is a promise the data cannot keep.

Seeded below; **DeerFlow v2 supplies the full set of compositions**. Each entry
is `{component_symbol: weight}`; weights are normalised at calculation time, so
they may be written as shares, percentages, or market-cap figures.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

#: symbol -> its components and their weights. Every component must be a symbol
#: the synthetic snapshot carries (see market_data/symbols.py INSTRUMENTS and
#: tests/fixtures/prices_synthetic.parquet), or the index cannot be materialised.
COMPOSITIONS: Mapping[str, Mapping[str, float]] = {
    # Vanguard Total World Stock: US total market and total international, at
    # roughly the market-cap split the fund itself holds.
    "VT": {"VTI": 0.60, "VXUS": 0.40},
}


def is_custom_index(symbol: str) -> bool:
    return symbol in COMPOSITIONS


def composition(symbol: str) -> Optional[Mapping[str, float]]:
    """The parts of `symbol`, or None if it is not a custom index."""
    return COMPOSITIONS.get(symbol)


def components(symbol: str) -> Tuple[str, ...]:
    """The instruments `symbol` is made of — the series the data service must
    deliver for the valuation engine to compute the index."""
    return tuple((COMPOSITIONS.get(symbol) or {}).keys())
