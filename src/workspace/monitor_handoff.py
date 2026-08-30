"""Save → **Monitor This Strategy**: hand a saved plan to wealth-manager as a durable
monitored portfolio, and send the user to the Portfolio Operations workspace.

This is the product loop's RAAAL half. RAAAL stays the research/evaluate/save entry;
when a person chooses to *live with* a strategy over time, this:

  1. builds a versioned :class:`SavedStrategyPlan` from the saved plan's **sealed**
     native intent (native ``intent_hash`` carried verbatim as ``source_intent_hash`` —
     the same chain-of-custody contract the runtime-artifact export already uses);
  2. hands its wire form to wealth-manager's ``POST /app/portfolios/monitor``, which
     instantiates a monitored portfolio (simulated holdings now; imported/linked later)
     and returns its ``portfolio_id`` + workspace ``scope``;
  3. yields the workspace URL that opens that portfolio's Portfolio Operations view.

The wealth-manager call is an **injectable seam** (:func:`set_client`) so the flow is
testable without a live wealth-manager, and so a deployment wires the real base URL +
service token via env. RAAAL never re-implements any portfolio logic — it hands over a
verified plan and navigates; wealth-manager owns the monitoring.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

#: the three holdings sources wealth-manager understands (deploy-now = SIMULATED).
HOLDINGS_SOURCES = ("SIMULATED", "IMPORTED", "LINKED")
DEFAULT_HOLDINGS_SOURCE = "SIMULATED"


class MonitorUnavailable(RuntimeError):
    """Wealth-manager is not configured/reachable — the handoff cannot complete."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _protocol_version() -> str:
    """Best-effort protocol version to stamp on the plan's provenance (wealth-manager
    verifies the plan's self-consistency, not this value)."""
    try:
        import runtime_contracts as rc
        return str(getattr(rc, "CONTRACT_VERSION", getattr(rc, "__version__", "")))
    except Exception:
        return ""


def _snapshot_id(reading: Any) -> str:
    for attr in ("market_data_snapshot_id", "snapshot_id", "market_snapshot_id"):
        val = getattr(reading, attr, None)
        if val:
            return str(val)
    return ""


def build_saved_plan(stored: dict, reading: Any, *, plan_id: str,
                     owner_id: str = "", tenant_id: str = "") -> Any:
    """Build a :class:`SavedStrategyPlan` from a reopened saved plan's sealed intent.

    Refuses a plan with no sealed intent (no ``intent_hash``) — there would be no
    identity to carry across the boundary, exactly as the runtime-artifact export
    refuses. The plan's ``content_hash`` is computed from its own meaning on
    construction, so it is byte-identical to what wealth-manager recomputes on import."""
    from .saved_strategy_plan import SavedStrategyPlan

    intent = getattr(reading, "intent", None)
    if intent is None or not getattr(intent, "intent_hash", None):
        raise ValueError("this plan has no sealed intent to monitor")
    now = _now()
    label = str(stored.get("text", "") or plan_id)
    return SavedStrategyPlan.from_intent(
        intent, label=label,
        methodology={"id": stored.get("picked", ""), "version": ""},
        protocol_version=_protocol_version(),
        market_data_snapshot_id=_snapshot_id(reading),
        created_at=now, effective_at=now,
        owner_id=owner_id, tenant_id=tenant_id, plan_id=plan_id, plan_version=1)


# ── the wealth-manager seam ──────────────────────────────────────────────────────
class WealthManagerClient(Protocol):
    def monitor(self, plan_dict: dict, *, holdings_source: str,
                owner_id: str) -> dict: ...


class HttpWealthManagerClient:
    """The real client — a thin ``POST /app/portfolios/monitor`` over httpx."""

    def __init__(self, base_url: str, token: str = "") -> None:
        self._base = base_url.rstrip("/")
        self._token = token

    def monitor(self, plan_dict: dict, *, holdings_source: str, owner_id: str) -> dict:
        import httpx
        bearer = self._token or f"dev:{owner_id or 'raaal'}"
        resp = httpx.post(
            f"{self._base}/app/portfolios/monitor",
            json={"saved_plan": plan_dict, "holdings_source": holdings_source},
            headers={"Authorization": f"Bearer {bearer}"}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


_CLIENT_OVERRIDE: Optional[WealthManagerClient] = None


def set_client(client: Optional[WealthManagerClient]) -> None:
    """Test/deploy seam: force the wealth-manager client (``None`` restores env-config)."""
    global _CLIENT_OVERRIDE
    _CLIENT_OVERRIDE = client


def _client() -> Optional[WealthManagerClient]:
    if _CLIENT_OVERRIDE is not None:
        return _CLIENT_OVERRIDE
    base = os.environ.get("WEALTH_MANAGER_BASE_URL", "")
    if not base:
        return None
    return HttpWealthManagerClient(
        base, os.environ.get("WEALTH_MANAGER_SERVICE_TOKEN", ""))


def workspace_url(portfolio_id: str, *, scope: str = "") -> str:
    """The Portfolio Operations URL that opens a monitored portfolio in the workspace."""
    base = os.environ.get("WORKSPACE_BASE_URL", "").rstrip("/")
    return f"{base}/app/plan/{portfolio_id}"


def monitor_plan(stored: dict, reading: Any, *, plan_id: str,
                 holdings_source: str = DEFAULT_HOLDINGS_SOURCE,
                 owner_id: str = "", tenant_id: str = "") -> dict:
    """Build the plan, hand it to wealth-manager, return its monitor result (with
    ``portfolio_id`` + workspace ``scope``). Raises :class:`MonitorUnavailable` when
    wealth-manager is not configured."""
    if holdings_source not in HOLDINGS_SOURCES:
        holdings_source = DEFAULT_HOLDINGS_SOURCE
    plan = build_saved_plan(stored, reading, plan_id=plan_id, owner_id=owner_id,
                            tenant_id=tenant_id)
    client = _client()
    if client is None:
        raise MonitorUnavailable(
            "wealth-manager base URL is not configured (WEALTH_MANAGER_BASE_URL)")
    result = client.monitor(plan.to_dict(), holdings_source=holdings_source,
                            owner_id=owner_id)
    result.setdefault("workspace_url", workspace_url(result.get("portfolio_id", "")))
    return result


__all__ = [
    "HOLDINGS_SOURCES", "DEFAULT_HOLDINGS_SOURCE", "MonitorUnavailable",
    "build_saved_plan", "WealthManagerClient", "HttpWealthManagerClient",
    "set_client", "workspace_url", "monitor_plan",
]
