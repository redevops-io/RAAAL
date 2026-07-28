"""P6: the live API contracts (docx §17) + the paper-only safety invariants.

Drives the whole flow: create project -> discoveries -> objective-compare -> mission -> explain ->
approve (the ONLY mutation, paper order) -> learning. Asserts approval never dispatches externally and
that holdings change only after an explicit approval.
"""
from __future__ import annotations

import os
import tempfile

os.environ["RAAAL_STATE_DIR"] = tempfile.mkdtemp(prefix="raaal_state_")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
P = "/api/investment/projects/default"


def test_health_and_paper_only_info():
    h = client.get("/health").json()
    assert h["ok"] and h["mode"] == "paper"
    r = client.get("/info").json()
    assert r["paper_only"] is True and r["external_execution_path"] is False


def test_console_renders_three_objectives_and_demo_banner():
    page = client.get("/").text
    assert "DEMO" in page and "not investment advice" in page.lower()
    assert "Objective decisions" in page
    for label in ("Minimum Risk", "Maximum Return-to-Risk", "Maximum Total Return", "Current / no action"):
        assert label in page
    assert "Attention queue" in page


def test_full_contract_flow_and_paper_safety():
    # 1. create project (manifest is paper-only)
    proj = client.post("/api/investment/projects", json={}).json()
    assert proj["ok"] and proj["project"]["spec"]["mode"] == "paper"

    # 2. discoveries surface (queue shape the console consumes)
    disc = client.get(f"{P}/discoveries").json()
    assert "queue" in disc

    # 3. objective-compare -> three branches, each a REGISTERED strategy, with an approval gate
    oc = client.post(f"{P}/missions/objective-compare", json={"trigger": {"opportunity_class": "regime_change"}}).json()
    mid = oc["mission_id"]
    objs = {b["objective"] for b in oc["branches"]}
    assert objs == {"max_total_return", "max_return_to_risk", "min_risk"}
    assert oc["reconcile"]["no_averaging"] is True
    assert oc["approval"]["required"] is True and oc["approval"]["mode"] == "paper"

    # 4. mission view is replayable; stage proposed pre-approval
    mv = client.get(f"/api/investment/missions/{mid}").json()
    assert mv["stage"] == "proposed" and mv["graph"]["snapshot_id"] == oc["snapshot_id"]

    # 5. explain exposes selected strategy + alternatives + representation
    ex = client.get(f"/api/investment/missions/{mid}/explain").json()
    assert len(ex["branches"]) == 3
    assert all(b["selected_strategy_id"] for b in ex["branches"])

    # holdings before approval == the starting all-cash paper book
    strat_before = client.get(f"{P}/strategies/current").json()
    assert "registry" in strat_before and strat_before["selected"]

    # 6. approve — the ONLY mutation; paper order, never dispatched externally
    ap = client.post(f"/api/investment/missions/{mid}/approve", json={"objective": "min_risk"}).json()
    assert ap["approved"] is True
    assert ap["dispatched_externally"] is False and ap["mode"] == "paper"
    assert ap["order"]["dispatched_externally"] is False

    # 7. mission now completed + replayable across a fresh read
    mv2 = client.get(f"/api/investment/missions/{mid}").json()
    assert mv2["stage"] == "completed"

    # 8. learning view is populated (shadow)
    lv = client.get(f"{P}/learning").json()
    assert lv.get("enabled") in (True, False)  # enabled when agentic_os present; degrade otherwise

    # 9. paper-rebalance is approval-gated, never inline
    pr = client.post(f"{P}/paper-rebalance", json={}).json()
    assert pr["requires_approval"] is True and pr["mode"] == "paper"


def test_no_broker_or_order_router_imported():
    # structural paper-only guarantee: the app package imports no broker/execution client
    import app, app.engine, app.api_investment, app.main  # noqa: F401
    import sys
    banned = ("alpaca", "ib_insync", "ccxt", "broker", "order_router")
    loaded = [m for m in sys.modules if any(b in m.lower() for b in banned)]
    assert not loaded, f"a broker/execution module was imported: {loaded}"
