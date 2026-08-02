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


@pytest.fixture(autouse=True)
def no_model_calls(request, monkeypatch):
    """The default suite never calls a language model.

    Stage 1 may use one, and the moment `anthropic` was installed alongside an
    `ANTHROPIC_API_KEY` the workspace tests started making live API calls — they
    became nondeterministic, network-dependent, billable, and dependent on which
    machine ran them. One promptly failed because the model raised a question
    the deterministic rules do not.

    Same rule as the licensed market data: reaching outside is opt-in, and a
    test that silently acquires a network dependency passes locally, fails in
    CI, and gets diagnosed as flaky for a week. Opt in with
    `@pytest.mark.model_stage1`.
    """
    if request.node.get_closest_marker("model_stage1"):
        return
    try:
        import src.workspace.routes as routes
    except Exception:                                           # pragma: no cover
        return
    monkeypatch.setattr(routes, "_parser_client", lambda: None, raising=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model_stage1: exercises model-assisted compiler stage 1 against a live "
        "API. Requires ANTHROPIC_API_KEY and network; opt-in, because the "
        "default suite must produce the same result on every machine.")
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


@pytest.fixture(autouse=True)
def _pilot_data_policy(monkeypatch):
    """Run the suite under the closed-pilot boundary, stated rather than
    inherited.

    `_prices()` fails closed without this, so a suite that did not declare a
    policy would quietly stop exercising every run path — and the journey tests
    would pass by producing nothing.
    """
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
