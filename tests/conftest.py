"""Shared fixtures, and the market-data tier split.

Two datasets, two purposes:

    tests/fixtures/prices_synthetic.parquet   committed, invented, no network
    the licensed snapshot                     private, immutable, pinned by hash

The default suite runs entirely on the first. It needs no credentials, reaches
no network, and produces the same numbers on a fresh clone as on the machine
that wrote it. That is the point: the repository, not one workstation, has to be
able to reproduce the application.

The licensed snapshot is for integration, benchmark and research runs, behind
`-m market_data_integration`. When that marker is not requested, missing
credentials are not a failure — nobody asked for the licensed data.

Nothing measured on the synthetic fixture is a claim about any real security. It
is shaped like market data so the evaluation stack has something realistic to
run on, and deliberately not calibrated to anything.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "prices_synthetic.parquet"

#: The vendor snapshot, when a developer has produced one locally. Used only by
#: the integration tier; the default suite ignores it even when present, so a
#: result cannot differ between two machines because one happened to have it.
LICENSED = REPO_ROOT / "data" / "history" / "prices.parquet"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data_integration: runs against the licensed snapshot. Requires "
        "credentials and network; fails rather than skips when explicitly "
        "requested, because a silent skip in the tier that exists to check the "
        "real data is indistinguishable from a pass.")


@pytest.fixture(scope="session")
def synthetic_prices():
    """The committed fixture, loaded through the ordinary loader.

    Deliberately not `pd.read_parquet` — going through the loader means the
    integrity check runs on every suite, so a fixture edited by hand is caught
    here rather than by a confusing assertion three layers up.
    """
    from src.market_data import load_prices, synthetic_snapshot

    return load_prices(synthetic_snapshot())


@pytest.fixture
def prices_on_disk(monkeypatch, tmp_path, synthetic_prices):
    """Point the web and workspace routes at the synthetic fixture.

    Both read a module-level path at request time. Redirecting that is what
    makes the rendered result panels — and therefore the pages asserting
    provenance is visible — work from a clone alone.
    """
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes

    monkeypatch.setattr(web_routes, "PRICES", SYNTHETIC)
    monkeypatch.setattr(workspace_routes, "PRICES", SYNTHETIC)
    return synthetic_prices


@pytest.fixture(scope="session")
def licensed_snapshot():
    """The pinned licensed snapshot, for the integration tier only."""
    from src.market_data import load_prices, production_snapshot

    snapshot = production_snapshot()
    return load_prices(snapshot, allow_network=True), snapshot


def requires_licensed_data(func):
    """Mark a test as belonging to the licensed tier."""
    return pytest.mark.market_data_integration(func)


#: Set by the integration tier to make absence a failure rather than a skip.
LICENSED_REQUIRED = os.environ.get("QUANTIFY_REQUIRE_MARKET_DATA") == "1"
