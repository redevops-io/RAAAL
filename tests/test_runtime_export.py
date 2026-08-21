"""The real export seam (#7): a native RAAAL intent → a canonical runtime artifact.

Exercises the wiring on RAAAL's ACTUAL intent (built by catalog_intent), not a
stand-in — the native intent_hash is carried verbatim as the source identity, the
canonical rcv1 runtime-artifact hash is distinct, and the shared-form payload is
extracted from the intent's own fields. Internals are untouched: this only reads a
sealed intent and emits the wire artifact.
"""
from __future__ import annotations

import pytest

import runtime_contracts as rc
from src.workspace.catalog_intent import intent_for
from src.workspace.runtime_export import payload_from_intent, runtime_artifact_for


def test_export_produces_dual_identity_from_a_real_intent():
    intent, _ = intent_for("stated-weights")
    assert intent is not None and intent.intent_hash

    art = runtime_artifact_for(intent, label="plan-x")

    # native source identity carried verbatim (never recomputed)
    assert art["source_intent_hash"] == intent.intent_hash
    # canonical runtime-artifact identity, recomputable from the payload
    assert art["runtime_artifact_hash"] == rc.content_hash(art["payload"])
    # the two are different kinds and never interchangeable
    assert art["source_intent_hash"] != art["runtime_artifact_hash"]
    assert art["provenance"]["source_runtime"] == "raaal"


def test_payload_extracts_the_shared_form_from_intent_fields():
    intent, _ = intent_for("stated-weights")
    p = payload_from_intent(intent, label="plan-x")
    assert p["allocation_method"] == "stated_weights"
    # "60/40" + "stocks,bonds" → fractional weights
    assert p["target_allocation"] == {"stocks": 0.6, "bonds": 0.4}
    assert p["universe"] == ["stocks", "bonds"]
    assert p["label"] == "plan-x"


def test_optimizer_strategy_states_no_explicit_weights():
    # a method-driven strategy (inverse_volatility) carries the method, not weights
    intent, _ = intent_for("risk-based-allocation")
    p = payload_from_intent(intent)
    assert p["allocation_method"] == "inverse_volatility"
    assert p["target_allocation"] == {}          # the method drives; consumer derives


def test_export_refuses_an_unsealed_intent():
    class Unsealed:
        intent_hash = None
        objective = ""
        fields: dict = {}
    with pytest.raises(ValueError):
        runtime_artifact_for(Unsealed())
