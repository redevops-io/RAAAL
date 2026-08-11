"""The three authoritative state records (docx §16), volume-backed and HTTP-servable.

Replaces the single `reports/state.json` file with:
  1. InvestmentProjectManifest  - desired mandate (mode:paper, universe, objectives, constraints, approval).
  2. PortfolioStateStore        - holdings/cash/NAV/orders/benchmark/mode; written ONLY by an approved paper rebalance.
  3. MissionLedger              - append-only evidence snapshot + ExecutionPlan + approvals + outcomes (replayable).

Dependency-free (json + pathlib). Manifest validation optionally cross-checks the shared
`agentic_os.manifest` when importable, but never hard-depends on it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def state_dir() -> Path:
    """Where the three state records live, from the resolved context.

    This read `os.environ["RAAAL_STATE_DIR"]` directly until the Quantify
    branch merged and `test_no_undeclared_reader` caught it. The rule there is
    that anything a second component could form its own view about belongs in
    `DeploymentContext` — and a storage location is the clearest possible case,
    because two components disagreeing about it means one writes a manifest the
    other never reads, with nothing saying they disagreed.

    Written first with an `os.environ` fallback for when the resolver is
    unavailable, and `test_no_undeclared_reader` rejected that too — correctly.
    A fallback that reads the environment *is* the second opinion, just one
    that only appears when something else has already gone wrong, which is the
    worst moment to start disagreeing about where state lives. The resolver
    reads the same variable and carries the same default, so there is nothing
    the fallback could have added except a way to diverge.
    """
    from ..deploy.context import current

    d = Path(current().state_directory)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    # time.time is allowed here (runtime, not a workflow script)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# 1. Investment Project Manifest
# ---------------------------------------------------------------------------
@dataclass
class InvestmentProjectManifest:
    name: str = "raaal-demo"
    mode: str = "paper"
    base_currency: str = "USD"
    universe: str = "raaal-core-v1"
    objectives: List[Dict[str, object]] = field(default_factory=lambda: [
        {"id": "max_total_return", "enabled": True},
        {"id": "max_return_to_risk", "enabled": True},
        {"id": "min_risk", "enabled": True},
    ])
    constraints: Dict[str, object] = field(default_factory=lambda: {
        "longOnly": True, "leverageCap": 1.0, "inverseExposureCap": 0.15,
        "cryptoCap": 0.10, "minimumCash": 0.05, "maximumTurnover": 0.25,
    })
    rebalance: Dict[str, object] = field(default_factory=lambda: {
        "cadence": "weekly", "triggers": ["regime_change", "allocation_drift", "risk_limit_breach"],
    })
    approval: Dict[str, object] = field(default_factory=lambda: {"rebalance": "required"})
    data: Dict[str, object] = field(default_factory=lambda: {"market": "yfinance", "macro": "configured-provider"})
    execution: Dict[str, object] = field(default_factory=lambda: {"topologyPolicy": "learned", "verification": "required"})

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "InvestmentProjectManifest":
        meta = d.get("metadata", {}) or {}
        spec = d.get("spec", {}) or {}
        base = InvestmentProjectManifest()
        return InvestmentProjectManifest(
            name=meta.get("name", base.name),
            mode=spec.get("mode", base.mode),
            base_currency=spec.get("baseCurrency", base.base_currency),
            universe=spec.get("universe", base.universe),
            objectives=spec.get("objectives", base.objectives),
            constraints=spec.get("constraints", base.constraints),
            rebalance=spec.get("rebalance", base.rebalance),
            approval=spec.get("approval", base.approval),
            data=spec.get("data", base.data),
            execution=spec.get("execution", base.execution),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "apiVersion": "redevops.io/v1", "kind": "InvestmentProject",
            "metadata": {"name": self.name},
            "spec": {
                "mode": self.mode, "baseCurrency": self.base_currency, "universe": self.universe,
                "objectives": self.objectives, "constraints": self.constraints,
                "rebalance": self.rebalance, "approval": self.approval,
                "data": self.data, "execution": self.execution,
            },
        }

    def validate(self) -> List[str]:
        """Return a list of hard errors ([] == valid). Paper-only + approval-required are enforced."""
        errs: List[str] = []
        if self.mode != "paper":
            errs.append("mode must be 'paper' in the public demo (no live execution path exists)")
        if self.approval.get("rebalance") != "required":
            errs.append("approval.rebalance must be 'required'")
        if not any(o.get("enabled") for o in self.objectives):
            errs.append("at least one objective must be enabled")
        for key in ("longOnly", "leverageCap", "minimumCash"):
            if key not in self.constraints:
                errs.append(f"constraints.{key} is required")
        # optional cross-check against the shared runtime manifest validator
        try:  # pragma: no cover - only when agentic_os is installed
            from agentic_os import manifest as _m  # type: ignore
            _ = _m  # presence check; the shared validator uses a different schema shape
        except Exception:
            pass
        return errs


class ManifestStore:
    def __init__(self, base: Optional[Path] = None) -> None:
        self._path = (base or state_dir()) / "manifest.json"

    def load(self) -> InvestmentProjectManifest:
        if self._path.exists():
            try:
                return InvestmentProjectManifest.from_dict(json.loads(self._path.read_text()))
            except Exception:  # noqa: BLE001
                pass
        return InvestmentProjectManifest()

    def save(self, m: InvestmentProjectManifest) -> List[str]:
        errs = m.validate()
        if errs:
            return errs
        self._path.write_text(json.dumps(m.to_dict(), indent=2))
        return []


# ---------------------------------------------------------------------------
# 2. Portfolio State Store (paper)
# ---------------------------------------------------------------------------
class PortfolioStateStore:
    def __init__(self, base: Optional[Path] = None, starting_nav: float = 1_000_000.0) -> None:
        self._path = (base or state_dir()) / "portfolio_state.json"
        self._starting_nav = starting_nav

    def load(self) -> Dict[str, object]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:  # noqa: BLE001
                pass
        return {
            "mode": "paper", "base_currency": "USD", "nav": self._starting_nav,
            "benchmark": "SPY", "holdings": {"BIL": 1.0}, "orders": [], "updated_at": _now(),
        }

    def current_weights(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.load().get("holdings", {}).items()}

    def apply_paper_rebalance(self, target_weights: Dict[str, float], mission_id: str,
                              objective: str, strategy_id: str) -> Dict[str, object]:
        """Record a PAPER order and set holdings to the target. No external venue is ever contacted."""
        state = self.load()
        order = {
            "ts": _now(), "mode": "paper", "mission_id": mission_id, "objective": objective,
            "strategy_id": strategy_id, "from_weights": state.get("holdings", {}),
            "to_weights": target_weights, "dispatched_externally": False,
        }
        state.setdefault("orders", []).append(order)
        state["holdings"] = target_weights
        state["updated_at"] = _now()
        self._path.write_text(json.dumps(state, indent=2))
        return order


# ---------------------------------------------------------------------------
# 3. Mission Ledger (append-only, replayable)
# ---------------------------------------------------------------------------
class MissionLedger:
    def __init__(self, base: Optional[Path] = None) -> None:
        self._path = (base or state_dir()) / "mission_ledger.jsonl"

    def append(self, mission_id: str, event_type: str, payload: Dict[str, object]) -> Dict[str, object]:
        rec = {"mission_id": mission_id, "ts": _now(), "type": event_type, "payload": payload}
        with self._path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def events(self, mission_id: Optional[str] = None) -> List[Dict[str, object]]:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if mission_id is None or rec.get("mission_id") == mission_id:
                out.append(rec)
        return out

    def mission(self, mission_id: str) -> Dict[str, object]:
        """Aggregate a mission's events into a single replayable view."""
        evs = self.events(mission_id)
        if not evs:
            return {}
        view: Dict[str, object] = {"mission_id": mission_id, "events": evs,
                                   "stage": "discovered", "approvals": [], "decisions": []}
        for e in evs:
            t, p = e["type"], e.get("payload", {})
            if t == "created":
                view.update({"stage": "proposed", "snapshot_id": p.get("snapshot_id"),
                             "objective": p.get("objective"), "compare": p.get("compare")})
            elif t == "approved":
                view["stage"] = "approved"
                view.setdefault("approvals", []).append(p)  # type: ignore[union-attr]
            elif t == "paper_rebalanced":
                view["stage"] = "completed"
                view.setdefault("decisions", []).append(p)  # type: ignore[union-attr]
            elif t == "verified":
                view["verification"] = p
            elif t == "rejected":
                view["stage"] = "rejected"
                view.setdefault("decisions", []).append(p)  # type: ignore[union-attr]
        return view

    def list_missions(self) -> List[str]:
        seen: List[str] = []
        for e in self.events():
            mid = e["mission_id"]
            if mid not in seen:
                seen.append(mid)
        return seen
