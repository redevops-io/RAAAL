"""P5: the objective_compare mission graph (3 selection branches, no averaging, approval gate)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agentic.objective_compare import build_objective_compare_mission
from src.agentic.selection import build_snapshot
from src.config import AUX_SERIES, OBJECTIVES, UNIVERSE
from src.features import compute_returns, exponential_cov, exponential_mean
from src.pipeline import compare_from_snapshot
from src.strategies import CAPABILITY_BY_ID

BASE = [a.ticker for a in UNIVERSE]


def _compare():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=420)
    cols = list(dict.fromkeys(BASE + AUX_SERIES + ["GLD", "SPY", "BIL"]))
    data = {}
    for t in cols:
        if t == "^VIX":
            data[t] = rng.uniform(12, 22, len(dates))
        elif t == "BIL":
            data[t] = 100 * (1 + 0.00008) ** np.arange(len(dates))
        else:
            data[t] = 100 * np.exp(np.cumsum(rng.normal(rng.uniform(-0.0002, 0.0006), rng.uniform(0.004, 0.02), len(dates))))
    prices = pd.DataFrame(data, index=dates)
    returns = compute_returns(prices)[BASE].dropna()
    snap = build_snapshot(prices, returns, "risk_on", exponential_mean(returns), exponential_cov(returns), rf=0.0)
    return compare_from_snapshot(snap)


def test_mission_has_three_selection_branches_no_averaging_and_a_gate():
    compare = _compare().to_dict()
    m = build_objective_compare_mission(compare, trigger={"opportunity_class": "regime_change"})
    assert m["kind"] == "objective_compare" and m["snapshot_id"] == compare["snapshot_id"]
    # exactly the three objective branches, each naming a REGISTERED strategy
    objs = {b["objective"] for b in m["branches"]}
    assert objs == set(OBJECTIVES)
    for b in m["branches"]:
        assert b["selected_strategy_id"] in CAPABILITY_BY_ID or b["selected_strategy_id"] == "no_action"
        assert "representation" in b  # each branch carries its Execution Planner representation choice
    # reconcile compares, never averages, and keeps the current/no-action baseline
    assert m["reconcile"]["no_averaging"] is True
    assert m["reconcile"]["current"]["label"].lower().startswith("current")
    assert len(m["reconcile"]["comparison"]) == len(OBJECTIVES)
    # approval gate: human, paper, required
    assert m["approval"] == {"required": True, "mode": "paper", "gate": "human",
                             "note": m["approval"]["note"]}
    # verification checks every branch's weights sum to one + registered
    assert any(c["check"] == "weights_sum_to_one" and c["ok"] for c in m["verify"]["checks"])


def test_context_bundle_ties_reps_to_snapshot():
    compare = _compare().to_dict()
    m = build_objective_compare_mission(compare)
    cb = m["context_bundle"]
    assert cb["snapshot_id"] == compare["snapshot_id"]
    assert set(cb["mandate_constraints"]) >= {"long_only", "minimum_cash", "inverse_exposure_cap"}
    # representation plans present for each objective with a chosen mix + candidates
    for obj in OBJECTIVES:
        rp = cb["representation_plans"].get(obj)
        assert rp and rp.get("chosen") and rp.get("candidates")
