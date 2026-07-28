"""Context Runtime: the versioned evidence bundle behind every recommendation.

Ties one snapshot (snapshot_id, regime, mandate constraints, universe) to the representation plan the
Execution Planner chose for each objective (which evidence/representation/lookback/depth), so every
ObjectivePlan can be traced back to both the numbers (snapshot) and the reasoning route (EXPLAIN).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from ..config import MANDATE_CONSTRAINTS, UNIVERSE


def _digest(obj: object) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:12]


def representation_plans(objectives: List[str]) -> Dict[str, dict]:
    """Per objective, the Execution Planner's representation decision + EXPLAIN (guarded import)."""
    try:
        from agentic_os.planner.domains import investment_investigation, investment_planner
    except Exception:  # pragma: no cover - runtime without agentic_os
        return {}
    planner = investment_planner()
    out: Dict[str, dict] = {}
    for obj in objectives:
        dfp = planner.plan(investment_investigation(obj, subject="raaal-demo"), min_confidence=0.9)
        out[obj] = dfp.explain()
    return out


def build_context_bundle(compare: dict, reps: Optional[Dict[str, dict]] = None) -> dict:
    """Assemble the evidence bundle from an ObjectiveCompareResult.to_dict()."""
    objectives = list((compare.get("plans") or {}).keys())
    reps = reps if reps is not None else representation_plans(objectives)
    return {
        "snapshot_id": compare.get("snapshot_id"),
        "as_of": compare.get("as_of"),
        "regime": compare.get("regime"),
        "regime_diagnostics": compare.get("regime_diagnostics", {}),
        "universe": [a.ticker for a in UNIVERSE],
        "mandate_constraints": dict(MANDATE_CONSTRAINTS),
        "mu_digest": _digest(compare.get("rf_rate")) if compare else "",
        "representation_plans": reps,   # objective -> Execution Planner EXPLAIN (chosen reps + candidates)
        "provenance": "context-runtime",
    }
