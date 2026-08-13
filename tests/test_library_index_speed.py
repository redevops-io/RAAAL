"""The research library index, which took 36 seconds on the deployment.

Reported as "research library is not opening when clicked on". It opened. A
browser that spins for half a minute is indistinguishable from one that is
stuck, and nobody waits to find out which.

The page calls `_evaluate_full` for every version of every concept, and each
call is a backtest — 1.9s per version, four versions. Everything else on the
page costs 0.3s together.

Caching a computation is only safe if the key carries everything the answer
depends on, so the first test here is not about speed at all: it renders the
page with the cache and against a cache that has been disabled, and requires
the two to be the same page.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api import app

    return TestClient(app)


def _library_html(client) -> str:
    response = client.get("/ui/")
    assert response.status_code == 200
    return response.text


class TestTheCacheChangesNothingButTheTime:
    def test_the_cached_page_equals_the_uncached_page(self, client,
                                                       monkeypatch):
        """The property that makes the cache admissible. If a memo can return
        a different page from a fresh evaluation, it is not a cache — it is a
        second implementation with its own answers."""
        import src.web.routes as routes

        routes._EVALUATIONS.clear()
        with_cache = _library_html(client)

        monkeypatch.setattr(routes, "_evaluate_full", routes._evaluate_uncached)
        without_cache = _library_html(client)

        assert with_cache == without_cache, (
            "the memoised page differs from the freshly evaluated one; the key "
            "is missing something the evaluation depends on")

    def test_a_recorded_trial_changes_the_key(self):
        """The ledger is in the key because it changes the verdict: the trial
        count scales the deflated Sharpe denominator. A cache keyed only on the
        version would serve yesterday's conclusion after a trial was recorded,
        which is the failure mode that makes caching dangerous here."""
        import src.web.routes as routes

        methodologies, protocols, _, _, ledger = routes._registries()
        concept = next(iter(methodologies.concepts()))
        version = methodologies.get(concept)
        protocol = routes._pick_protocol(version, protocols)

        class MovedOn:
            def trial_breakdown(self, concept):
                return {"attempted_trials": 99, "dsr_countable_trials": 99}

            def list_errata(self):
                return []

        before = routes._evaluation_key(version, protocol, ledger)
        after = routes._evaluation_key(version, protocol, MovedOn())
        assert before != after


class TestThePageIsUsable:
    #: Generous on purpose. The deployment runs a t3.small and this suite runs
    #: wherever it runs; what is being caught is a page that re-runs every
    #: backtest per request, which is seconds rather than milliseconds out.
    BUDGET = 2.0

    def test_a_warm_request_is_not_a_backtest(self, client):
        _library_html(client)  # warm

        started = time.monotonic()
        _library_html(client)
        elapsed = time.monotonic() - started

        assert elapsed < self.BUDGET, (
            f"the library index took {elapsed:.1f}s on a warm process. It is "
            "re-running evaluations that have not changed, and on the "
            "deployment's instance that reads as a link that does not open")
