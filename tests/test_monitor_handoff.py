"""Save → **Monitor This Strategy** — the RAAAL half of the monitored-portfolio loop.

RAAAL builds a versioned ``SavedStrategyPlan`` from a saved plan's sealed intent and
hands its wire form to wealth-manager, then navigates the user to the Portfolio
Operations workspace. These tests hold the load-bearing properties without a live
wealth-manager (the client is an injectable seam):

  * the handed-over plan is a **self-consistent** ``SavedStrategyPlan`` (its
    ``content_hash`` matches its own meaning — so wealth-manager, running the identical
    ``rcv1`` recipe, accepts it) carrying the RAAAL ``source_intent_hash`` verbatim;
  * a successful handoff redirects to the workspace URL for the new portfolio;
  * the chosen holdings source is passed through;
  * when wealth-manager isn't configured, the flow degrades gracefully (no error page,
    the save/evaluate surface is untouched).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.workspace import monitor_handoff, pilot_routes, pilot_store
from src.workspace.catalog_intent import intent_for


class _Reading:
    intent = None


@pytest.fixture
def sealed_reading():
    intent, _ = intent_for("stated-weights")
    assert intent is not None and intent.intent_hash
    r = _Reading()
    r.intent = intent
    return r


@pytest.fixture
def client(monkeypatch, sealed_reading):
    from src import api

    monkeypatch.setattr(pilot_routes, "deployment_uses_the_runtime", lambda: True)
    monkeypatch.setattr(pilot_store, "load",
                        lambda plan_id: {"plan_id": plan_id, "picked": "stated-weights",
                                         "text": "60/40 VTI/BND"})
    monkeypatch.setattr(pilot_routes, "reopen", lambda stored: sealed_reading)

    async def _no_csrf(request):        # CSRF handled elsewhere; not under test here
        return None
    monkeypatch.setattr(pilot_routes, "_csrf_refusal", _no_csrf)
    monkeypatch.setattr("src.workspace.owner.current", lambda: "user-a")
    yield TestClient(api.app)
    monitor_handoff.set_client(None)    # never leak the stub across tests


# ═══════════════════════════════════════════════════════════════════════════════
# 1. build_saved_plan — a self-consistent plan carrying the native seal verbatim
# ═══════════════════════════════════════════════════════════════════════════════
def test_build_saved_plan_is_self_consistent(sealed_reading):
    plan = monitor_handoff.build_saved_plan(
        {"picked": "stated-weights", "text": "60/40"}, sealed_reading,
        plan_id="plan-x", owner_id="user-a")
    wire = plan.to_dict()
    assert wire["schema"] == "raaal/saved-strategy-plan"
    # the native seal is carried verbatim as the chain-of-custody head
    assert wire["provenance"]["source_intent_hash"] == sealed_reading.intent.intent_hash
    # the content_hash is computed from the plan's own meaning — wealth-manager runs the
    # identical rcv1 recipe, so a self-consistent plan here is an importable plan there
    from src.workspace.saved_strategy_plan import SavedStrategyPlan
    assert SavedStrategyPlan.from_dict(wire).content_hash == wire["content_hash"]


def test_build_saved_plan_refuses_an_unsealed_intent():
    r = _Reading()
    r.intent = None
    with pytest.raises(ValueError):
        monitor_handoff.build_saved_plan({}, r, plan_id="p")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. the route — hands over the plan and redirects to the workspace
# ═══════════════════════════════════════════════════════════════════════════════
def test_monitor_hands_over_plan_and_redirects(client, monkeypatch):
    captured: dict = {}

    class _FakeWM:
        def monitor(self, plan_dict, *, holdings_source, owner_id):
            captured["plan"] = plan_dict
            captured["holdings_source"] = holdings_source
            captured["owner_id"] = owner_id
            return {"portfolio_id": "rcv1:abc123", "scope": "household:mp-x"}

    monitor_handoff.set_client(_FakeWM())
    monkeypatch.setenv("WORKSPACE_BASE_URL", "https://workspace.example")

    r = client.post("/pilot/plans/plan-x/monitor",
                    data={"holdings_source": "SIMULATED"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "https://workspace.example/app/plan/rcv1:abc123"

    # a valid, self-consistent plan was handed over with the source seal + chosen source
    assert captured["plan"]["schema"] == "raaal/saved-strategy-plan"
    assert captured["plan"]["provenance"]["source_intent_hash"]
    assert captured["holdings_source"] == "SIMULATED"
    assert captured["owner_id"] == "user-a"


def test_holdings_source_choice_is_passed_through(client, monkeypatch):
    seen: dict = {}

    class _FakeWM:
        def monitor(self, plan_dict, *, holdings_source, owner_id):
            seen["hs"] = holdings_source
            return {"portfolio_id": "rcv1:xyz"}

    monitor_handoff.set_client(_FakeWM())
    client.post("/pilot/plans/plan-x/monitor", data={"holdings_source": "IMPORTED"},
                follow_redirects=False)
    assert seen["hs"] == "IMPORTED"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. graceful degradation when wealth-manager isn't configured
# ═══════════════════════════════════════════════════════════════════════════════
def test_monitor_degrades_when_wealth_manager_unconfigured(client, monkeypatch):
    monitor_handoff.set_client(None)
    monkeypatch.delenv("WEALTH_MANAGER_BASE_URL", raising=False)
    r = client.post("/pilot/plans/plan-x/monitor",
                    data={"holdings_source": "SIMULATED"}, follow_redirects=False)
    assert r.status_code == 503
    assert "not configured" in r.text
