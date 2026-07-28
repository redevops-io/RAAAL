"""P2: one snapshot -> three objective plans, and the three volume-backed state records."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agentic.selection import build_snapshot
from src.agentic.state_store import (
    InvestmentProjectManifest,
    ManifestStore,
    MissionLedger,
    PortfolioStateStore,
)
from src.config import AUX_SERIES, OBJECTIVES, UNIVERSE
from src.features import compute_returns, exponential_cov, exponential_mean
from src.pipeline import compare_from_snapshot

BASE = [a.ticker for a in UNIVERSE]


def _snapshot(regime: str = "risk_on"):
    rng = np.random.default_rng(3)
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
    returns = compute_returns(prices)
    base_returns = returns[BASE].dropna()
    return build_snapshot(prices, base_returns, regime, exponential_mean(base_returns),
                          exponential_cov(base_returns), rf=0.0)


def test_three_plans_share_one_snapshot():
    snap = _snapshot()
    res = compare_from_snapshot(snap)
    assert set(res.plans) == set(OBJECTIVES)
    for plan in res.plans.values():
        assert plan.evidence_snapshot_id == snap.snapshot_id  # every recommendation ties to ONE bundle
    assert res.current["label"].lower().startswith("current")
    d = res.to_dict()
    assert d["snapshot_id"] == snap.snapshot_id and set(d["plans"]) == set(OBJECTIVES)


def test_manifest_is_paper_only(tmp_path):
    store = ManifestStore(base=tmp_path)
    m = InvestmentProjectManifest()
    assert m.validate() == []           # default is valid
    assert store.save(m) == []
    reloaded = store.load()
    assert reloaded.mode == "paper" and reloaded.approval["rebalance"] == "required"
    # a live-mode manifest is rejected
    bad = InvestmentProjectManifest(mode="live")
    errs = store.save(bad)
    assert any("paper" in e for e in errs)


def test_paper_rebalance_never_dispatches_externally(tmp_path):
    store = PortfolioStateStore(base=tmp_path)
    target = {"SPY": 0.5, "TLT": 0.3, "BIL": 0.2}
    order = store.apply_paper_rebalance(target, mission_id="m1", objective="min_risk", strategy_id="risk_parity")
    assert order["dispatched_externally"] is False and order["mode"] == "paper"
    assert store.current_weights() == target


def test_mission_ledger_replays(tmp_path):
    led = MissionLedger(base=tmp_path)
    led.append("m1", "created", {"snapshot_id": "snap_x", "objective": "min_risk", "compare": {}})
    led.append("m1", "approved", {"by": "operator", "objective": "min_risk"})
    led.append("m1", "paper_rebalanced", {"strategy_id": "risk_parity"})
    # a fresh store reads the same append-only log (replayable across restarts)
    view = MissionLedger(base=tmp_path).mission("m1")
    assert view["stage"] == "completed"
    assert view["snapshot_id"] == "snap_x"
    assert len(view["events"]) == 3
    assert "m1" in MissionLedger(base=tmp_path).list_missions()
