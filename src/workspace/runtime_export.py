"""Export a native RAAAL intent as a canonical runtime artifact (dual identity).

This is the real export seam #7: it wires the boundary adapter
(`src.runtime_boundary.to_runtime_artifact`) to RAAAL's actual intent. RAAAL's
native IntentField model is NOT migrated — this only READS a finalized intent and
emits the wire artifact a downstream runtime (wealth-manager) consumes.

The two identities stay distinct and are never interchanged:
  * source_intent_hash    — the native `intent.intent_hash`, carried verbatim;
  * runtime_artifact_hash — the canonical `rcv1` hash of the extracted payload,
                            computed by the adapter.

RAAAL owns extracting the shared-form payload from its own intent (below); the
adapter owns only the wire mechanics. This module reaches into intent semantics;
the adapter never does.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..runtime_boundary import to_runtime_artifact

# The intent fields RAAAL resolves that map onto the portable strategy selection a
# downstream finance runtime needs.
_ALLOCATION_METHOD = "allocation_method"
_STATED_WEIGHTS = "stated_weights"
_ASSETS = "assets"
_CADENCE = "cadence"


def _field(intent: Any, name: str, default: str = "") -> str:
    field = getattr(intent, "fields", {}).get(name)
    if field is None:
        return default
    value = getattr(field, "value", field)
    return "" if value is None else str(value)


def _target_allocation(intent: Any) -> dict[str, float]:
    """Parse RAAAL's stated weights + assets into fractional weights, e.g.
    ``"60/40"`` + ``"stocks,bonds"`` → ``{"stocks": 0.6, "bonds": 0.4}``. Returns
    ``{}`` when the strategy states no explicit weights (an optimizer method like
    inverse_volatility) — the allocation_method then carries the meaning, and the
    consumer derives the weights."""
    weights_raw = _field(intent, _STATED_WEIGHTS)
    assets_raw = _field(intent, _ASSETS)
    if not weights_raw or not assets_raw:
        return {}
    try:
        parts = [float(w) for w in weights_raw.replace("-", "/").split("/") if w.strip()]
        assets = [a.strip() for a in assets_raw.split(",") if a.strip()]
    except ValueError:
        return {}
    if not parts or len(parts) != len(assets):
        return {}
    total = sum(parts)
    if total <= 0:
        return {}
    return {asset: weight / total for asset, weight in zip(assets, parts)}


def payload_from_intent(intent: Any, *, label: str = "") -> dict:
    """The shared-form strategy selection extracted from a native intent."""
    universe = tuple(a.strip() for a in _field(intent, _ASSETS).split(",") if a.strip())
    return {
        "allocation_method": _field(intent, _ALLOCATION_METHOD),
        "target_allocation": _target_allocation(intent),
        "objective": getattr(intent, "objective", ""),
        "rebalancing": _field(intent, _CADENCE),
        "universe": list(universe),
        "label": label,
    }


def runtime_artifact_for(intent: Any, *, label: str = "") -> dict:
    """A native intent → its canonical runtime artifact (dual identity).

    Refuses an intent that is not sealed (has no `intent_hash`): there would be no
    native identity to carry, and an unsealed intent is not something another
    runtime should build a policy on.
    """
    source = getattr(intent, "intent_hash", None)
    if not source:
        raise ValueError(
            "cannot export an unsealed intent — it has no intent_hash to carry as "
            "the source identity across the boundary")
    return to_runtime_artifact(
        source_intent_hash=source,
        payload=payload_from_intent(intent, label=label))
