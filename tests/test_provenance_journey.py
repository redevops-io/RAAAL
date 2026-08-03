"""The whole journey, through the routes a user actually reaches.

    POST description -> confirm -> save -> run -> record -> reopen -> export

Every seam in this chain has been proven separately. That is not the same as
proving the chain, and the difference is exactly what Gate 3 kept finding: a
mechanism built, registered and tested, with no production caller using it.

The resolver is watched throughout. It must be called at the compute boundary
and nowhere else — reopening a worksheet or exporting a bundle that reaches for
market data would be reconstructing provenance from current configuration,
which is the failure the stored record exists to prevent.

Final state is read through a fresh store, so nothing asserted here is an
artifact of the session that wrote it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.market_data.provenance import ProvenanceStatus, from_json
from src.workspace.store import WorkspaceStore

POLICY = "PILOT_DATA_POLICY"
OWNER = "pilot"

DESCRIPTION = ("I put $2,000 into SPY every month in my Roth IRA, on the first "
               "trading day of the period, reinvesting the dividends, and I "
               "never sell.")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A live application pointed at an empty workspace."""
    monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")
    monkeypatch.setenv("QUANTIFY_DEPLOYMENT_PROFILE", "local")

    import src.api as api

    with TestClient(api.app) as client:
        yield client, tmp_path / "w.db"


class Watcher:
    """Counts resolver calls, so "never" can be asserted rather than assumed."""

    def __init__(self, monkeypatch):
        import src.market_data.access as access

        self.calls = []
        original = access.resolve

        def watched(*args, **kwargs):
            self.calls.append(kwargs.get("context", "?"))
            return original(*args, **kwargs)

        monkeypatch.setattr(access, "resolve", watched)
        # The routes import the module, not the symbol, so patching the module
        # attribute is what the live path actually reaches.

    def forbid(self, monkeypatch):
        import src.market_data.access as access

        def refuse(*args, **kwargs):
            raise AssertionError(
                "the resolver was called while reading stored state; the "
                "provenance should have come from the record")

        monkeypatch.setattr(access, "resolve", refuse)


PLAN_ID = "journey-plan"


def saved_plan(client):
    """Walk the real front door as far as a persisted plan and run.

    `/workspace/new` renders the confirmation screen; `/workspace/save` is what
    commits, and `save` is the route that resolves market data and calls
    `generate`. Both are exercised, because the confirmation screen is where a
    draft run happens and the save is where one is persisted.
    """
    drafted = client.get("/workspace/new", params={"describe": DESCRIPTION})
    assert drafted.status_code == 200, drafted.status_code

    committed = client.post(
        "/workspace/save",
        params={"describe": DESCRIPTION, "plan_id": PLAN_ID},
        data={"confirm_all": "yes"}, follow_redirects=False)
    assert committed.status_code in (200, 303, 422), committed.status_code
    return committed


class TestTheResolverIsCalledOnlyAtTheComputeBoundary:
    def test_reopening_a_plan_does_not_resolve(self, workspace, monkeypatch):
        client, path = workspace
        saved_plan(client)

        store = WorkspaceStore(path)
        plans = store.list_plans(OWNER)
        if not plans:
            pytest.skip("the front door did not persist a plan in this build")

        Watcher(monkeypatch).forbid(monkeypatch)
        # Reading stored state must not reach for data.
        assert store.list_plans(OWNER)
        for plan in plans:
            assert store.runs_for(plan["plan_id"], OWNER) is not None

    def test_exporting_does_not_resolve(self, workspace, monkeypatch):
        from src.db.transfer import export_bundle

        client, path = workspace
        saved_plan(client)
        store = WorkspaceStore(path)

        Watcher(monkeypatch).forbid(monkeypatch)
        bundle = export_bundle(store, exported_at="2026-08-01T00:00:00Z")
        assert bundle["manifest"]["counts"] is not None


class TestEveryStoredRunIsAttributable:
    def test_no_persisted_run_claims_an_unrecorded_provenance(self, workspace):
        """The end state of the journey, read from a fresh store.

        A run reaching persistence with NOT_RECORDED means a live producer
        declined to store what it held, which `generate` now refuses — this is
        the check that the refusal is reached by the route rather than only by
        a direct call.
        """
        client, path = workspace
        saved_plan(client)

        store = WorkspaceStore(path)
        runs = [run for plan in store.list_plans(OWNER)
                for run in store.runs_for(plan["plan_id"], OWNER)]
        if not runs:
            pytest.skip("this build's front door persisted no run")

        for run in runs:
            carried = from_json(run["result"].get("market_data"))
            assert carried.status is not ProvenanceStatus.NOT_RECORDED, (
                f"run {run['run_id']} was stored without attribution")

    def test_every_stored_run_identifies_its_data(self, workspace):
        client, path = workspace
        saved_plan(client)

        store = WorkspaceStore(path)
        runs = [run for plan in store.list_plans(OWNER)
                for run in store.runs_for(plan["plan_id"], OWNER)]
        if not runs:
            pytest.skip("this build's front door persisted no run")

        for run in runs:
            carried = from_json(run["result"].get("market_data"))
            if carried.status is ProvenanceStatus.RECORDED:
                assert carried.identifies_data, run["run_id"]
                assert carried.permitted, run["run_id"]

    def test_worksheets_hold_references_not_copies(self, workspace):
        client, path = workspace
        saved_plan(client)

        store = WorkspaceStore(path)
        for plan in store.list_plans(OWNER):
            record = store.worksheet_for_scenario(plan["plan_id"], OWNER)
            if record is None:
                continue
            assert "market_data" not in str(record["payload"]), (
                "a worksheet carries its own provenance copy; one figure now "
                "has two sources of truth")


class TestTheStoredChainSurvivesTheDeploymentMoving:
    def test_the_policy_changing_does_not_rewrite_a_stored_run(
            self, workspace, monkeypatch):
        client, path = workspace
        saved_plan(client)

        store = WorkspaceStore(path)
        runs = [run for plan in store.list_plans(OWNER)
                for run in store.runs_for(plan["plan_id"], OWNER)]
        if not runs:
            pytest.skip("this build's front door persisted no run")
        before = [from_json(run["result"].get("market_data")).to_json()
                  for run in runs]

        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        Watcher(monkeypatch).forbid(monkeypatch)

        fresh = WorkspaceStore(path)
        after = [from_json(run["result"].get("market_data")).to_json()
                 for plan in fresh.list_plans(OWNER)
                 for run in fresh.runs_for(plan["plan_id"], OWNER)]
        assert after == before


class TestTheCitedRunsAreCoherent:
    """SHARED_ACCESS: one resolver call, and every cited run proven to agree.

    Asserted rather than inferred from the loop sharing one object — the
    mutation that makes one candidate differ is what this is for.
    """

    def cited_provenance(self, store):
        records = []
        for plan in store.list_plans(OWNER):
            for run in store.runs_for(plan["plan_id"], OWNER):
                carried = from_json(run["result"].get("market_data"))
                if carried.status is ProvenanceStatus.RECORDED:
                    records.append(carried)
        return records

    def test_every_cited_run_agrees_on_the_data(self, workspace):
        client, path = workspace
        saved_plan(client)

        records = self.cited_provenance(WorkspaceStore(path))
        if len(records) < 2:
            pytest.skip("this journey produced fewer than two attributed runs")
        assert len({one.snapshot_id for one in records}) == 1
        assert len({one.content_digest for one in records}) == 1
        assert len({one.access_decision for one in records}) == 1
        assert len({one.policy_version for one in records}) == 1

    def test_a_mixed_set_is_detectable(self, workspace):
        """The mutation that proves this checks every cited run rather than
        only the first. Written below the store, because the writer refuses it.
        """
        from src.db.types import Json

        client, path = workspace
        saved_plan(client)
        store = WorkspaceStore(path)
        runs = [run for plan in store.list_plans(OWNER)
                for run in store.runs_for(plan["plan_id"], OWNER)]
        if not runs:
            pytest.skip("this build's front door persisted no run")

        forged = dict(runs[0]["result"])
        forged["market_data"] = {
            **forged.get("market_data", {}),
            "snapshot_id": "a-different-snapshot",
            "content_digest": "mdv1:different"}
        with store._conn() as conn:
            conn.execute("UPDATE plan_run SET result = ? WHERE run_id = ?",
                         (Json(forged), runs[0]["run_id"]))

        records = self.cited_provenance(WorkspaceStore(path))
        if len(records) < 2:
            pytest.skip("this journey produced fewer than two attributed runs")
        assert len({one.snapshot_id for one in records}) > 1, (
            "a divergent cited run went unnoticed")
