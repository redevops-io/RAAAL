"""PortfolioEngine: the in-process orchestration behind the API.

Network-free by default: builds the evidence snapshot from the nightly parquet history
(data/history/prices.parquet) and falls back to a small deterministic synthetic series so the service
runs anywhere. Holds the three state records + the discovery + selection + learning wiring.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.agentic.discovery_runtime import InvestmentDiscovery
from src.agentic.objective_compare import build_objective_compare_mission
from src.agentic.selection import Snapshot, build_snapshot
from src.agentic.signals import observation_from
from src.agentic.state_store import ManifestStore, MissionLedger, PortfolioStateStore
from src.config import AUX_SERIES, UNIVERSE
from src.features import compute_returns, exponential_cov, exponential_mean
from src.pipeline import compare_from_snapshot
from src.portfolio_utils import portfolio_metrics, rf_from_sgov
from src.strategies import strategy_capabilities

BASE = [a.ticker for a in UNIVERSE]
PRICES_PATH = Path(os.environ.get("RAAAL_PRICES_PATH", "data/history/prices.parquet"))


def _synthetic_prices(seed: int = 5, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    cols = list(dict.fromkeys(BASE + AUX_SERIES + ["GLD", "SPY", "BIL"]))
    data: Dict[str, np.ndarray] = {}
    for t in cols:
        if t == "^VIX":
            data[t] = rng.uniform(12, 22, n)
        elif t == "BIL":
            data[t] = 100 * (1 + 0.00008) ** np.arange(n)
        else:
            data[t] = 100 * np.exp(np.cumsum(rng.normal(rng.uniform(-0.0002, 0.0006), rng.uniform(0.004, 0.02), n)))
    return pd.DataFrame(data, index=dates)


class PortfolioEngine:
    def __init__(self) -> None:
        self.manifests = ManifestStore()
        self.portfolio = PortfolioStateStore()
        self.ledger = MissionLedger()
        self.discovery = InvestmentDiscovery()
        self._learn = None
        self._compare_cache: Optional[dict] = None
        self._missions: Dict[str, dict] = {}   # mission_id -> mission graph (also mirrored in the ledger)

    # --- evidence snapshot -------------------------------------------------------------------
    def _prices(self) -> pd.DataFrame:
        if PRICES_PATH.exists():
            try:
                df = pd.read_parquet(PRICES_PATH)
                if not df.empty:
                    return df
            except Exception:  # noqa: BLE001
                pass
        return _synthetic_prices()

    def snapshot(self) -> Snapshot:
        prices = self._prices()
        returns = compute_returns(prices)
        base = returns[[t for t in BASE if t in returns.columns]].dropna()
        try:
            from src.regime import detect_regime
            regime = detect_regime(prices, returns).name
        except Exception:  # noqa: BLE001
            regime = "risk_on"
        mu, cov, rf = exponential_mean(base), exponential_cov(base), rf_from_sgov(prices)
        return build_snapshot(prices, base, regime, mu, cov, rf)

    def _current(self, snap: Snapshot) -> dict:
        w = self.portfolio.current_weights() or {"BIL": 1.0}
        return {"label": "Current / no action", "weights": w,
                "metrics": portfolio_metrics(w, snap.mu, snap.cov, snap.rf)}

    def compare(self, refresh: bool = False) -> dict:
        if self._compare_cache is None or refresh:
            snap = self.snapshot()
            res = compare_from_snapshot(snap, current=self._current(snap), last_regime=None)
            self._compare_cache = res.to_dict()
        return self._compare_cache

    # --- discovery ---------------------------------------------------------------------------
    def discoveries(self) -> dict:
        obs = observation_from(self.compare(), self.portfolio.load())
        obs.prev_regime = None    # a live regime-change signal would come from the prior cycle's store
        return self.discovery.cycle(obs)

    # --- missions ----------------------------------------------------------------------------
    def create_objective_compare(self, trigger: Optional[dict] = None) -> dict:
        compare = self.compare(refresh=True)
        mission = build_objective_compare_mission(compare, trigger=trigger)
        self._missions[mission["mission_id"]] = mission
        self.ledger.append(mission["mission_id"], "created",
                           {"snapshot_id": mission["snapshot_id"], "objective": "objective_compare",
                            "mission": mission})
        return mission

    def mission(self, mission_id: str) -> dict:
        view = self.ledger.mission(mission_id)
        graph = self._missions.get(mission_id) or (view.get("compare") if view else None)
        if not graph:
            # reconstruct the graph from the ledger's created event
            for e in self.ledger.events(mission_id):
                if e["type"] == "created":
                    graph = e["payload"].get("mission")
                    break
        if graph:
            view["graph"] = graph
        return view

    def explain(self, mission_id: str) -> dict:
        m = self.mission(mission_id).get("graph") or {}
        return {"mission_id": mission_id, "snapshot_id": m.get("snapshot_id"),
                "branches": [{"objective": b["objective"], "selected_strategy_id": b["selected_strategy_id"],
                              "alternative_strategy_ids": b["alternative_strategy_ids"],
                              "representation": b.get("representation"), "explain": b.get("explain"),
                              "score_table": b.get("score_table", [])} for b in m.get("branches", [])]}

    def approve(self, mission_id: str, objective: str = "max_return_to_risk", by: str = "operator") -> dict:
        """The ONLY state-mutating action. Applies a PAPER order for the chosen objective's plan."""
        m = self.mission(mission_id).get("graph")
        if not m:
            return {"approved": False, "error": "unknown mission"}
        branch = next((b for b in m["branches"] if b["objective"] == objective), None)
        if branch is None:
            return {"approved": False, "error": f"no branch for objective {objective}"}
        self.ledger.append(mission_id, "approved", {"by": by, "objective": objective})
        order = self.portfolio.apply_paper_rebalance(
            branch["weights"], mission_id=mission_id, objective=objective,
            strategy_id=branch["selected_strategy_id"])
        self.ledger.append(mission_id, "paper_rebalanced",
                           {"objective": objective, "strategy_id": branch["selected_strategy_id"],
                            "order": order})
        return {"approved": True, "dispatched_externally": False, "mode": "paper", "order": order}

    def reject(self, mission_id: str, reason: str = "", by: str = "operator") -> dict:
        self.ledger.append(mission_id, "rejected", {"by": by, "reason": reason})
        return {"rejected": True}

    def paper_rebalance(self, objective: str = "max_return_to_risk") -> dict:
        """Create an approval-gated mission (never executes inline)."""
        mission = self.create_objective_compare(trigger={"opportunity_class": "manual_rebalance"})
        return {"mission_id": mission["mission_id"], "requires_approval": True, "mode": "paper",
                "objective": objective, "note": "A paper rebalance requires human approval; nothing was executed."}

    # --- strategies + learning ---------------------------------------------------------------
    def strategies_current(self) -> dict:
        compare = self.compare()
        selected = {obj: plan.get("selected_strategy_id") for obj, plan in (compare.get("plans") or {}).items()}
        return {"registry": strategy_capabilities(), "selected": selected, "regime": compare.get("regime")}

    def learning(self) -> dict:
        if self._learn is None:
            try:
                from agentic_os.learning.bindings import investment as inv
                rt = inv.build_runtime(shadow=True)
                inv.seed_demo(rt, rounds=80)
                self._learn = rt
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                return {"enabled": False, "reason": str(exc)[:120]}
        return self._learn.learning_view()
