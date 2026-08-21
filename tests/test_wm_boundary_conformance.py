"""Cross-runtime conformance: a RAAAL runtime artifact is accepted by wealth-manager's verifier.

Proves the dual-identity boundary end to end — RAAAL's ``to_runtime_artifact`` output passes
wealth-manager's ``verify_runtime_artifact`` (fail-closed) and round-trips into a ``RaaalSelection``
with the native ``source_intent_hash`` preserved distinctly from the canonical identity. Skips if
wealth-manager is not importable (it is present only in a conformance environment); the boundary
invariants themselves are covered locally by ``test_runtime_boundary.py``.
"""
from __future__ import annotations

import pytest

import runtime_contracts as rc
from src.runtime_boundary import to_runtime_artifact

pytest.importorskip("wealth_manager.contracts_runtime")
pytest.importorskip("wealth_manager.discovery.from_raaal")
from wealth_manager.contracts_runtime import verify_runtime_artifact  # noqa: E402
from wealth_manager.discovery.from_raaal import from_runtime_artifact  # noqa: E402

NATIVE = rc.content_hash({"native_intent": "risk-parity", "seq": 1})
PAYLOAD = {"allocation_method": "risk_parity",
           "target_allocation": {"SPY": 0.6, "TLT": 0.4}, "objective": "growth"}


def test_raaal_artifact_passes_wm_verify_and_preserves_source():
    art = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    verify_runtime_artifact(art)                       # fail-closed; raises if the contract is violated
    sel = from_runtime_artifact(art)
    assert sel.source_intent_hash == NATIVE            # native identity preserved into the WM chain
    assert sel.source_intent_hash != art["runtime_artifact_hash"]   # kept distinct from the canonical id
    assert sel.allocation_method == "risk_parity"


def test_wm_rejects_a_tampered_payload():
    art = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    art["payload"]["allocation_method"] = "tampered"   # runtime_artifact_hash no longer matches payload
    with pytest.raises(Exception):
        verify_runtime_artifact(art)
