"""One request, end to end, through the deployed configuration.

    startup -> resolve target -> save -> run -> record -> reopen -> export

Every seam here was proven separately, and the chain still had two defects that
only appeared when a real request ran against a real PostgreSQL database:

    `WorkspaceStore()` never read `QUANTIFY_DATABASE_URL`, so the preflight
    validated PostgreSQL and the application wrote to a local SQLite file

    `worksheet_for_scenario` used SQLite's `json_extract`, which PostgreSQL
    does not have, so the query failed on every deployed request

Both passed every unit test. Neither was a wrong belief about a component; both
were two correct components disagreeing about what they were doing. That is
what a journey test is for, and it is why this one exists rather than more
seam tests.

**It fails by observing absent rows.** Restoring the store defect must make
this test notice that PostgreSQL holds nothing, not merely that two URLs
differ — the URL comparison lives in `test_store_target.py`, and a journey that
only repeated it would prove nothing new.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.market_data.provenance import ProvenanceStatus, from_json

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="the journey runs against the deployed engine; SQLite cannot "
           "evidence that the configured target is the one opened")

OWNER = "pilot"
PLAN_ID = "journey-plan"
LOCAL_FALLBACK = Path("data/workspace.db")

DESCRIPTION = ("I put $2,000 into SPY every month in my Roth IRA, on the first "
               "trading day of the period, reinvesting the dividends, and I "
               "never sell.")


def migrated():
    from sqlalchemy import text

    from src.db import migrate
    from src.db.engine import Database

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(database)
    return database


@pytest.fixture
def journey(monkeypatch):
    """Walk the real front door with the deployment pointed at PostgreSQL."""
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_DEPLOYMENT_PROFILE", "local")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", POSTGRES_URL)
    migrated()

    import src.api as api

    with TestClient(api.app) as client:
        drafted = client.get("/workspace/new", params={"describe": DESCRIPTION})
        assert drafted.status_code == 200, drafted.status_code
        saved = client.post(
            "/workspace/save",
            params={"describe": DESCRIPTION, "plan_id": PLAN_ID},
            data={"confirm_all": "yes"}, follow_redirects=False)
        assert saved.status_code == 303, (
            f"the front door did not commit: {saved.status_code}")
    return saved


def fresh_store():
    """A store of its own, so nothing here is an artifact of the request."""
    from src.workspace.store import WorkspaceStore

    return WorkspaceStore(POSTGRES_URL)


def stored_runs(store):
    return [run for plan in store.list_plans(OWNER)
            for run in store.runs_for(plan["plan_id"], OWNER)]


def forbid_resolver(monkeypatch):
    import src.market_data.access as access

    def refuse(*args, **kwargs):
        raise AssertionError(
            "the resolver was called while reading stored state; the "
            "provenance should have come from the record")

    monkeypatch.setattr(access, "resolve", refuse)
    monkeypatch.setattr(access, "resolve_prices", refuse)


class TestTheRequestReachedTheConfiguredDatabase:
    """The regression that interrupted this test being written."""

    def test_the_plan_is_in_postgresql(self, journey):
        store = fresh_store()
        assert store.db.dialect.value == "postgresql"
        assert [plan["plan_id"] for plan in store.list_plans(OWNER)] == [PLAN_ID]

    def test_a_run_is_in_postgresql(self, journey):
        assert stored_runs(fresh_store()), (
            "no run reached the configured database; the request wrote "
            "somewhere else")

    def test_a_worksheet_is_in_postgresql(self, journey):
        store = fresh_store()
        assert store.worksheet_for_scenario(PLAN_ID, OWNER) is not None, (
            "no worksheet reached the configured database — this query used "
            "SQLite's `json_extract`, which PostgreSQL does not have, and "
            "failed on every deployed request")

    def test_nothing_material_went_to_the_local_fallback(self, journey):
        """Where the rows went when the store ignored its configuration."""
        if not LOCAL_FALLBACK.exists():
            return
        conn = sqlite3.connect(LOCAL_FALLBACK)
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM plan WHERE plan_id = ?",
                (PLAN_ID,)).fetchone()[0]
        except sqlite3.OperationalError:
            return                       # no such table: nothing was written
        finally:
            conn.close()
        assert rows == 0, (
            f"{rows} row(s) for this journey are in the local SQLite fallback. "
            "The request wrote to a database the deployment did not configure")


class TestEveryStoredFigureIsAttributable:
    def test_no_run_claims_an_unrecorded_provenance(self, journey):
        for run in stored_runs(fresh_store()):
            carried = from_json(run["result"].get("market_data"))
            assert carried.status is ProvenanceStatus.RECORDED, (
                f"{run['run_id']} was stored without attribution; a live "
                "producer may not claim historical absence")

    def test_every_run_identifies_its_data(self, journey):
        for run in stored_runs(fresh_store()):
            carried = from_json(run["result"].get("market_data"))
            assert carried.identifies_data, run["run_id"]
            assert carried.content_digest, run["run_id"]

    def test_every_run_carries_an_allowed_decision(self, journey):
        for run in stored_runs(fresh_store()):
            carried = from_json(run["result"].get("market_data"))
            assert carried.permitted, run["run_id"]
            assert carried.access_decision is not None

    def test_all_seven_fields_are_present(self, journey):
        for run in stored_runs(fresh_store()):
            carried = from_json(run["result"].get("market_data"))
            assert carried.snapshot_id
            assert carried.content_digest
            assert carried.content_digest_version
            assert carried.license_class
            assert carried.policy_version
            assert carried.access_decision
            assert carried.accessed_at

    def test_the_cited_runs_agree_on_the_data(self, journey):
        """SHARED_ACCESS: one resolution, and every cited run proven to agree
        rather than assumed to because a loop shared an object."""
        records = [from_json(run["result"].get("market_data"))
                   for run in stored_runs(fresh_store())]
        assert records
        assert len({one.snapshot_id for one in records}) == 1
        assert len({one.content_digest for one in records}) == 1
        assert len({one.access_decision for one in records}) == 1
        assert len({one.policy_version for one in records}) == 1

    def test_the_worksheet_holds_a_reference_not_a_copy(self, journey):
        record = fresh_store().worksheet_for_scenario(PLAN_ID, OWNER)
        assert record is not None
        assert "market_data" not in str(record["payload"]), (
            "the worksheet carries its own provenance copy; one figure now has "
            "two sources of truth")
        assert record["payload"]["primary_run_ref"]


class TestReadingStoredStateResolvesNothing:
    def test_reopening_does_not_call_the_resolver(self, journey, monkeypatch):
        store = fresh_store()
        forbid_resolver(monkeypatch)
        assert store.list_plans(OWNER)
        assert store.worksheet_for_scenario(PLAN_ID, OWNER) is not None
        assert stored_runs(store)

    def test_exporting_does_not_call_the_resolver(self, journey, monkeypatch):
        from src.db.transfer import export_bundle

        store = fresh_store()
        forbid_resolver(monkeypatch)
        bundle = export_bundle(store, exported_at="2026-08-01T00:00:00Z")
        assert bundle["manifest"]["counts"]["plan_run"] >= 1

    def test_the_export_carries_the_stored_provenance(self, journey,
                                                      monkeypatch):
        from src.db.transfer import export_bundle

        store = fresh_store()
        before = [from_json(run["result"].get("market_data")).snapshot_id
                  for run in stored_runs(store)]
        forbid_resolver(monkeypatch)
        bundle = export_bundle(store, exported_at="2026-08-01T00:00:00Z")
        carried = [from_json(run["result"].get("market_data")).snapshot_id
                   for run in bundle["records"]["plan_run"]]
        assert carried == before


class TestTheChainSurvivesTheDeploymentMoving:
    def test_a_policy_change_does_not_rewrite_a_stored_run(self, journey,
                                                           monkeypatch):
        before = [from_json(run["result"].get("market_data")).to_json()
                  for run in stored_runs(fresh_store())]
        assert before

        monkeypatch.setenv("PILOT_DATA_POLICY",
                           "market-data-egress/pilot-vendor-approved@1")
        forbid_resolver(monkeypatch)

        after = [from_json(run["result"].get("market_data")).to_json()
                 for run in stored_runs(fresh_store())]
        assert after == before


class TestAMixedProvenanceSetIsVisible:
    """Proves the journey checks every cited run, not only the first."""

    def test_a_divergent_run_breaks_the_agreement(self, journey):
        from src.db.types import Json

        store = fresh_store()
        runs = stored_runs(store)
        assert runs

        forged = dict(runs[0]["result"])
        forged["market_data"] = {**forged["market_data"],
                                 "snapshot_id": "a-different-snapshot",
                                 "content_digest": "mdv1:different"}
        # Below the store: the writer would refuse this, which is the point.
        with store._conn() as conn:
            conn.execute("UPDATE plan_run SET result = ? WHERE run_id = ?",
                         (Json(forged), runs[0]["run_id"]))

        records = [from_json(run["result"].get("market_data"))
                   for run in stored_runs(fresh_store())]
        assert "a-different-snapshot" in {one.snapshot_id for one in records}, (
            "a divergent cited run went unnoticed")
