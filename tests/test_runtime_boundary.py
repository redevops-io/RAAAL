"""The RAAAL → runtime boundary adapter: dual identity, invariants held.

RAAAL's native intent model is not migrated; this adapter is the only crossing.
It must carry the native source identity verbatim, compute a distinct canonical
runtime-artifact identity, and refuse to emit an artifact that collides the two.
"""
from __future__ import annotations

import pytest

import runtime_contracts as rc
from src.runtime_boundary import BoundaryError, to_runtime_artifact, to_runtime_artifact_from_intent

NATIVE = rc.content_hash({"native_intent": "risk-parity", "seq": 1})
PAYLOAD = {"allocation_method": "risk_parity",
           "target_allocation": {"SPY": 0.6, "TLT": 0.4}, "objective": "growth"}


def test_emits_dual_identity_carrying_the_native_hash_verbatim():
    art = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    assert art["source_intent_hash"] == NATIVE                     # verbatim, not recomputed
    assert art["runtime_artifact_hash"] == rc.content_hash(PAYLOAD)
    assert art["source_intent_hash"] != art["runtime_artifact_hash"]  # different kinds
    assert art["protocol"]["canonicalization_version"] == rc.CANONICALIZATION_VERSION
    assert art["provenance"]["source_runtime"] == "raaal"


def test_refuses_a_missing_native_identity():
    with pytest.raises(BoundaryError):
        to_runtime_artifact(source_intent_hash="", payload=PAYLOAD)


def test_refuses_native_equal_to_runtime_identity():
    # if a caller passed the payload's own canonical hash as the "native" hash,
    # the two identities would be interchangeable — refused
    collide = rc.content_hash(PAYLOAD)
    with pytest.raises(BoundaryError):
        to_runtime_artifact(source_intent_hash=collide, payload=PAYLOAD)


def test_from_intent_pulls_the_native_intent_hash():
    class FakeIntent:
        intent_hash = NATIVE
    art = to_runtime_artifact_from_intent(FakeIntent(), PAYLOAD)
    assert art["source_intent_hash"] == NATIVE


def test_makes_the_runtime_contracts_version_explicit(monkeypatch):
    # freeze §6.1: a consumer must read the producing runtime-contracts version
    # directly, never infer it from the hash prefix
    import importlib.metadata as m
    art = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    assert art["protocol"]["runtime_contracts_version"] == m.version("runtime-contracts")


def test_carries_the_producer_build_revision_when_present(monkeypatch):
    monkeypatch.setenv("QUANTIFY_BUILD_COMMIT", "c8b5a8b")
    art = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    assert art["provenance"]["producer_version"] == "c8b5a8b"
    monkeypatch.delenv("QUANTIFY_BUILD_COMMIT")
    art2 = to_runtime_artifact(source_intent_hash=NATIVE, payload=PAYLOAD)
    assert art2["provenance"]["producer_version"] == ""   # empty, never fabricated
