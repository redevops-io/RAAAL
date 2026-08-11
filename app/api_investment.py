"""The investment API contracts (docx §17).

Approval is the ONLY state-mutating route, and it applies a PAPER order (no external venue). Everything
else is read-only. The project id is accepted for contract fidelity; the demo runs a single 'default'
project.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body

from src.agentic.state_store import InvestmentProjectManifest

router = APIRouter(prefix="/api/investment", tags=["investment"])


def _engine():
    from .engine import PortfolioEngine  # lazy so the module imports even if deps are missing
    if not hasattr(_engine, "_singleton"):
        _engine._singleton = PortfolioEngine()  # type: ignore[attr-defined]
    return _engine._singleton  # type: ignore[attr-defined]


# --- projects (manifest) -------------------------------------------------------------------
@router.post("/projects")
def create_project(body: dict = Body(default={})):
    eng = _engine()
    manifest = InvestmentProjectManifest.from_dict(body) if body else InvestmentProjectManifest()
    errors = eng.manifests.save(manifest)
    return {"ok": not errors, "errors": errors, "project": manifest.to_dict()}


@router.get("/projects/{project_id}")
def get_project(project_id: str):
    return _engine().manifests.load().to_dict()


# --- discoveries ---------------------------------------------------------------------------
@router.get("/projects/{project_id}/discoveries")
def discoveries(project_id: str):
    return _engine().discoveries()


# --- objective-compare mission -------------------------------------------------------------
@router.post("/projects/{project_id}/missions/objective-compare")
def objective_compare(project_id: str, body: dict = Body(default={})):
    mission = _engine().create_objective_compare(trigger=body.get("trigger"))
    return {"mission_id": mission["mission_id"], "stage": mission["stage"],
            "snapshot_id": mission["snapshot_id"], "regime": mission["regime"],
            "branches": [{"objective": b["objective"], "selected_strategy_id": b["selected_strategy_id"],
                          "weights": b["weights"], "benchmark_metrics": b["benchmark_metrics"],
                          "expected_tradeoffs": b["expected_tradeoffs"],
                          "binding_constraints": b["binding_constraints"],
                          "alternative_strategy_ids": b["alternative_strategy_ids"],
                          "representation": b.get("representation")} for b in mission["branches"]],
            "reconcile": mission["reconcile"], "approval": mission["approval"]}


@router.get("/missions/{mission_id}")
def get_mission(mission_id: str):
    return _engine().mission(mission_id)


@router.get("/missions/{mission_id}/explain")
def explain_mission(mission_id: str):
    return _engine().explain(mission_id)


@router.post("/missions/{mission_id}/approve")
def approve_mission(mission_id: str, body: dict = Body(default={})):
    return _engine().approve(mission_id, objective=body.get("objective", "max_return_to_risk"),
                             by=body.get("by", "operator"))


@router.post("/missions/{mission_id}/reject")
def reject_mission(mission_id: str, body: dict = Body(default={})):
    return _engine().reject(mission_id, reason=body.get("reason", ""), by=body.get("by", "operator"))


# --- strategies + learning + paper-rebalance -----------------------------------------------
@router.get("/projects/{project_id}/strategies/current")
def strategies_current(project_id: str):
    return _engine().strategies_current()


@router.get("/projects/{project_id}/learning")
def learning(project_id: str):
    return _engine().learning()


@router.post("/projects/{project_id}/paper-rebalance")
def paper_rebalance(project_id: str, body: dict = Body(default={})):
    return _engine().paper_rebalance(objective=body.get("objective", "max_return_to_risk"))
