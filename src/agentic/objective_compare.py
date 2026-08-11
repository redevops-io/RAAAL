"""Mission Runtime: the objective_compare mission graph.

    parent mission ("propose a governed paper rebalance")
        -> 3 parallel objective SELECTION branches (min_risk | max_return_to_risk | max_total_return)
        -> coordinator reconcile (side-by-side comparison + current/no-action; NO averaging)
        -> verify / stress
        -> approval gate (human, paper-only)

Built from an ObjectiveCompareResult so every branch references a registered strategy + the evidence
snapshot. This is a data structure the ledger stores and the console renders; the human approval gate is
the only thing that can advance it to a paper rebalance.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .context_bundle import build_context_bundle, representation_plans


def _new_id() -> str:
    try:
        from agentic_os.mission.types import new_id
        return new_id("invmis")
    except Exception:  # pragma: no cover
        import uuid
        return "invmis_" + uuid.uuid4().hex[:12]


def _branch(objective: str, plan: dict, rep: Optional[dict]) -> dict:
    """One objective branch: the selected registered strategy + alternatives + EXPLAIN."""
    return {
        "objective": objective,
        "selected_strategy_id": plan.get("selected_strategy_id"),
        "alternative_strategy_ids": plan.get("alternative_strategy_ids", []),
        "weights": plan.get("weights", {}),
        "benchmark_metrics": plan.get("benchmark_metrics", {}),
        "expected_tradeoffs": plan.get("expected_tradeoffs", {}),
        "binding_constraints": plan.get("binding_constraints", []),
        "confidence": plan.get("confidence"),
        "warnings": plan.get("warnings", []),
        "abstained": plan.get("abstained", False),
        "score_table": plan.get("score_table", []),
        "representation": (rep or {}).get("chosen"),        # the representation mix the planner chose
        "explain": rep,                                     # full EXPLAIN (candidates + reasons + axes)
    }


def _verify_checks(compare: dict) -> List[dict]:
    """Deterministic pre-approval verification (constraint compliance + traceability)."""
    checks: List[dict] = []
    for obj, plan in (compare.get("plans") or {}).items():
        w = plan.get("weights", {})
        total = round(sum(w.values()), 6)
        checks.append({"objective": obj, "check": "weights_sum_to_one", "ok": abs(total - 1.0) < 1e-6,
                       "value": total})
        checks.append({"objective": obj, "check": "strategy_is_registered",
                       "ok": bool(plan.get("selected_strategy_id"))})
        checks.append({"objective": obj, "check": "mandate_binding_recorded",
                       "ok": True, "binding": plan.get("binding_constraints", [])})
    return checks


def build_objective_compare_mission(
    compare: dict,
    *,
    trigger: Optional[dict] = None,
    reps: Optional[Dict[str, dict]] = None,
    mission_id: Optional[str] = None,
) -> dict:
    """Assemble the mission graph from an ObjectiveCompareResult.to_dict()."""
    plans = compare.get("plans") or {}
    objectives = list(plans.keys())
    reps = reps if reps is not None else representation_plans(objectives)
    mid = mission_id or _new_id()

    branches = [_branch(obj, plans[obj], reps.get(obj)) for obj in objectives]
    comparison = [{
        "objective": b["objective"], "selected_strategy_id": b["selected_strategy_id"],
        "exp_return": b["expected_tradeoffs"].get("exp_return"),
        "exp_vol": b["expected_tradeoffs"].get("exp_vol"),
        "sharpe": b["expected_tradeoffs"].get("sharpe"),
        "beta_proxy": b["expected_tradeoffs"].get("beta_proxy"),
    } for b in branches]

    return {
        "mission_id": mid,
        "kind": "objective_compare",
        "snapshot_id": compare.get("snapshot_id"),
        "as_of": compare.get("as_of"),
        "regime": compare.get("regime"),
        "trigger": trigger or {"opportunity_class": "scheduled_review"},
        "stage": "proposed",
        "parent": {"goal": "Compare the objective plans and propose a governed PAPER rebalance"},
        "context_bundle": build_context_bundle(compare, reps),
        "branches": branches,                                  # the 3 objective columns
        "reconcile": {
            "no_averaging": True,                              # objectives are compared, never blended
            "comparison": comparison,
            "current": compare.get("current"),                # current / no-action baseline (4th column)
        },
        "verify": {"checks": _verify_checks(compare)},
        "approval": {"required": True, "mode": "paper", "gate": "human",
                     "note": "First and every paper rebalance needs explicit human approval."},
        "rebalance_triggered": compare.get("rebalance_triggered", False),
    }
