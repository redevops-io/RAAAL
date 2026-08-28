"""The versioned ``SavedStrategyPlan`` — the handoff contract to Wealth Manager.

Gate 3: a versioned export whose golden fixture is the cross-repo contract. These
tests hold the invariants the wealth-manager import side depends on — identity is
the strategy meaning plus the pinned methodology/protocol/data and nothing else,
the native source seal is carried verbatim, broker authority can never be smuggled
in, and the committed fixture's ``content_hash`` is stable bytes both repos agree on
because both compute the same runtime-contracts ``rcv1`` hash.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import runtime_contracts as rc
from src.runtime_boundary import to_runtime_artifact as boundary_artifact
from src.workspace.catalog_intent import intent_for
from src.workspace.saved_strategy_plan import (
    ForbiddenAuthorityError,
    SavedStrategyPlan,
    SchemaVersionError,
    migrate,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "saved_strategy_plan.v1.json"
FIXTURE_CONTENT_HASH = (
    "rcv1:637f5976d20650b41e12b2198f08a831ba05f064360501ee82cc54a7ef756eab")


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _sealed_intent():
    intent, _ = intent_for("stated-weights")
    assert intent is not None and intent.intent_hash
    return intent


def _plan(**over):
    """A plan built from the real sealed intent, with overridable envelope."""
    intent = _sealed_intent()
    kw = dict(
        label="60/40 VTI/BND",
        methodology={"id": "quantify-core", "version": "m3"},
        protocol_version="p1",
        market_data_snapshot_id="prices-synthetic-20260819",
        evaluation_result_refs=[{"run_id": "run-1", "run_hash": "rcv1:eval-1"}],
        strategy_constraints={"excluded_securities": ["TSLA"]},
        created_at="2026-08-19T15:04:05+00:00",
        effective_at="2026-08-19T15:04:05+00:00",
        owner_id="owner-9f8e7d6c",
        tenant_id="tenant-1234",
        plan_id="plan-a1b2c3d4e5f6",
    )
    kw.update(over)
    return SavedStrategyPlan.from_intent(intent, **kw)


# --- round-trip -------------------------------------------------------------

def test_to_dict_from_dict_is_identity():
    plan = _plan()
    assert SavedStrategyPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_golden_fixture_round_trips_and_hash_matches_committed_value():
    data = _fixture()
    plan = SavedStrategyPlan.from_dict(data)
    # the committed bytes are exactly what to_dict() produces again
    assert plan.to_dict() == data
    # and the identity is the frozen, cross-repo value
    assert plan.content_hash == FIXTURE_CONTENT_HASH
    assert data["content_hash"] == FIXTURE_CONTENT_HASH


def test_fixture_content_hash_is_the_rcv1_hash_of_its_canonical_meaning():
    # a wealth-manager import with NO shared code, only the shared contract,
    # recomputes the identical hash — this asserts the exact recipe.
    data = _fixture()
    p = data["provenance"]
    canonical = {
        "schema_version": data["schema_version"],
        "verified_strategy_intent": data["verified_strategy_intent"],
        "strategy_constraints": data["strategy_constraints"],
        "methodology_id": p["methodology_id"],
        "methodology_version": p["methodology_version"],
        "protocol_version": p["protocol_version"],
        "market_data_snapshot_id": p["market_data_snapshot_id"],
    }
    assert rc.content_hash(canonical) == FIXTURE_CONTENT_HASH


# --- meaning-only identity --------------------------------------------------

def test_envelope_does_not_change_identity():
    base = _plan()
    other = _plan(owner_id="someone-else", tenant_id="tenant-999",
                  plan_id="plan-zzz", created_at="2020-01-01T00:00:00+00:00",
                  effective_at="2020-01-01T00:00:00+00:00",
                  evaluation_result_refs=[{"run_id": "run-2", "run_hash": "rcv1:eval-2"}])
    # same strategy meaning + same pinned methodology/protocol/data → same identity
    assert other.content_hash == base.content_hash
    # amendments and disclosure ack are envelope too
    amended = base.amend(note="clerical", at="2026-08-20T00:00:00+00:00")
    assert amended.content_hash == base.content_hash


def test_changed_methodology_protocol_or_data_changes_identity():
    base = _plan()
    assert _plan(methodology={"id": "quantify-core", "version": "m4"}).content_hash \
        != base.content_hash
    assert _plan(protocol_version="p2").content_hash != base.content_hash
    assert _plan(market_data_snapshot_id="prices-other").content_hash \
        != base.content_hash


def test_changed_strategy_meaning_changes_identity():
    base = _plan()
    moved = base.amend(note="rebalance to annual", at="2026-08-20T00:00:00+00:00",
                       verified_strategy_intent={**base.verified_strategy_intent,
                                                 "rebalancing": "annual"})
    assert moved.content_hash != base.content_hash


# --- source carried verbatim ------------------------------------------------

def test_source_intent_hash_is_carried_verbatim():
    intent = _sealed_intent()
    plan = _plan()
    assert plan.provenance["source_intent_hash"] == intent.intent_hash
    # and it is NOT the plan's own identity — different kinds
    assert plan.provenance["source_intent_hash"] != plan.content_hash


def test_unsealed_intent_is_refused():
    class Unsealed:
        intent_hash = None
        objective = ""
        fields: dict = {}

    with pytest.raises(ValueError):
        SavedStrategyPlan.from_intent(
            Unsealed(), methodology={"id": "m", "version": "1"},
            protocol_version="p1", market_data_snapshot_id="s",
            created_at="t", effective_at="t")


# --- forbidden keys ---------------------------------------------------------

@pytest.mark.parametrize("forbidden", [
    {"brokerage_credentials": {"api_key": "sk-live"}},
    {"execution_authorization": True},
    {"tax_lots": [{"symbol": "VTI", "qty": 3}]},
    {"household_restrictions": ["no crypto"]},
    {"suitability": "aggressive-ok"},
])
def test_forbidden_authority_refused_on_construction(forbidden):
    with pytest.raises(ForbiddenAuthorityError):
        _plan(strategy_constraints={"excluded_securities": ["TSLA"], **forbidden})


def test_forbidden_authority_refused_on_deserialization():
    data = _fixture()
    data["verified_strategy_intent"]["execution_authorization"] = True
    with pytest.raises(ForbiddenAuthorityError):
        SavedStrategyPlan.from_dict(data)


def test_forbidden_authority_refused_when_nested_deep():
    with pytest.raises(ForbiddenAuthorityError):
        _plan(strategy_constraints={"limits": {"broker": {"credentials": "x"}}})


# --- amend ------------------------------------------------------------------

def test_amend_bumps_version_records_prior_and_recomputes():
    base = _plan()
    amended = base.amend(note="tighten exclusion", at="2026-08-21T00:00:00+00:00",
                         strategy_constraints={"excluded_securities": ["TSLA", "GME"]})
    assert amended.plan_version == base.plan_version + 1
    assert amended.amendments[-1]["prior_content_hash"] == base.content_hash
    assert amended.amendments[-1]["plan_version"] == amended.plan_version
    assert amended.amendments[-1]["note"] == "tighten exclusion"
    # meaning-changing amendment → new identity
    assert amended.content_hash != base.content_hash
    # the base is untouched (frozen, returns a new plan)
    assert base.plan_version == 1 and base.amendments == ()


# --- dual-identity bridge ---------------------------------------------------

def test_to_runtime_artifact_bridges_to_the_existing_boundary():
    plan = _plan()
    art = plan.to_runtime_artifact()
    # native source carried verbatim through the bridge
    assert art["source_intent_hash"] == plan.provenance["source_intent_hash"]
    # canonical runtime-artifact identity = rcv1 hash of the shared-form payload
    assert art["runtime_artifact_hash"] == rc.content_hash(art["payload"])
    assert art["source_intent_hash"] != art["runtime_artifact_hash"]
    assert art["provenance"]["source_runtime"] == "raaal"
    # identical to what the boundary produces directly for the same inputs
    direct = boundary_artifact(source_intent_hash=plan.provenance["source_intent_hash"],
                               payload=dict(plan.verified_strategy_intent))
    assert art["runtime_artifact_hash"] == direct["runtime_artifact_hash"]


# --- versioning -------------------------------------------------------------

def test_unknown_or_newer_schema_version_is_refused():
    data = _fixture()
    data["schema_version"] = "v2"
    with pytest.raises(SchemaVersionError):
        migrate(data)
    with pytest.raises(SchemaVersionError):
        SavedStrategyPlan.from_dict(data)


def test_current_version_migrates_to_itself():
    data = _fixture()
    assert migrate(data)["schema_version"] == "v1"


def test_tampered_content_hash_is_refused():
    data = _fixture()
    data["content_hash"] = "rcv1:" + "0" * 64
    with pytest.raises(ValueError):
        SavedStrategyPlan.from_dict(data)
