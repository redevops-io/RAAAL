"""What a test result says about market data.

`record_run` refuses a result with no `market_data`, because `None` would mean
"market-derived", "not market-derived" and "unknown" at once and a reader could
not tell them apart. A test that stores a run and does not care about market
data therefore has to say which of the three it means, rather than omitting the
field and inheriting whichever interpretation was convenient.

Most of them mean the second: the figure is a fixture, not something priced
from a snapshot.
"""
from src.market_data.provenance import not_applicable, not_recorded

#: For a run whose result was constructed by a test rather than computed from
#: market data.
NO_MARKET_DATA = not_applicable("a fixture result, not priced from a snapshot")

#: For a run standing in for one written before provenance existed.
UNRECORDED_MARKET_DATA = not_recorded("a legacy row, predating provenance")
