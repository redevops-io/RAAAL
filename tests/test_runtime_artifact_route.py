"""The runtime-artifact export route as a boundary (freeze plan §6.2).

The artifact is content-addressed, so its `runtime_artifact_hash` is a strong
ETag: a consumer that already holds this exact artifact revalidates with
`If-None-Match` and gets 304 — never a stale body under a matching tag. Stubs the
store seams so the route's conditional-read logic is exercised without a database.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.workspace import pilot_routes, pilot_store
from src.workspace.catalog_intent import intent_for


@pytest.fixture
def client(monkeypatch):
    from src import api

    intent, _ = intent_for("stated-weights")
    assert intent is not None and intent.intent_hash

    class _Reading:
        pass
    reading = _Reading()
    reading.intent = intent

    # the deployment declares the runtime; the plan exists; reopening yields the
    # sealed intent — the three seams the route needs, without a database
    monkeypatch.setattr(pilot_routes, "deployment_uses_the_runtime", lambda: True)
    monkeypatch.setattr(pilot_store, "load", lambda plan_id: {"plan_id": plan_id})
    monkeypatch.setattr(pilot_routes, "reopen", lambda stored: reading)
    return TestClient(api.app)


def test_export_carries_the_identity_as_a_strong_etag(client):
    r = client.get("/pilot/plans/plan-x/runtime-artifact")
    assert r.status_code == 200
    body = r.json()
    assert r.headers["etag"] == f'"{body["runtime_artifact_hash"]}"'
    # the enriched metadata is present on the wire (freeze §6.1)
    assert body["protocol"]["runtime_contracts_version"]
    assert "producer_version" in body["provenance"]


def test_if_none_match_revalidates_to_304(client):
    first = client.get("/pilot/plans/plan-x/runtime-artifact")
    etag = first.headers["etag"]
    again = client.get("/pilot/plans/plan-x/runtime-artifact",
                       headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.headers["etag"] == etag
    assert again.content == b""            # no body re-sent


def test_a_different_etag_still_gets_the_body(client):
    r = client.get("/pilot/plans/plan-x/runtime-artifact",
                   headers={"If-None-Match": '"sha256:not-this-one"'})
    assert r.status_code == 200
    assert r.json()["runtime_artifact_hash"]
