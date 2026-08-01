"""Shared fixtures, and the one thing a clone cannot bring with it.

Market data is not in the repository. It is vendor-licensed and 350KB of daily
prices, and committing it would make a licensing decision on the project's
behalf that has not been taken — data-provider licensing is an open item for the
pilot review.

The consequence is that five tests exercising the rendered result panel have
nothing to render. They are skipped with a reason naming the command that
produces the data, rather than failing: a fresh clone reporting five confusing
failures teaches a new reader that the suite is broken, when what is true is
that one input is absent and reproducible in one command.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PRICE_HISTORY = Path("data/history/prices.parquet")

NO_PRICES = (
    "no local price history at data/history/prices.parquet. It is vendor data "
    "and deliberately not committed; regenerate it with "
    "`python3 -m src.history --start 2015-01-01 --end $(date +%F) --step 5`"
)

#: Apply to any test that needs a *computed result* rather than an artifact.
#: Everything in the artifact, methodology, policy and mission layers runs from
#: the repository alone; only the evaluated numbers need market data.
requires_price_history = pytest.mark.skipif(
    not PRICE_HISTORY.exists(), reason=NO_PRICES)


@pytest.fixture(scope="session")
def price_history_available() -> bool:
    return PRICE_HISTORY.exists()
