"""The end-to-end journey: describe, confirm, run, get a saved worksheet.

    describe -> confirm -> save -> run persisted -> worksheet revision 1
                                                 -> reopen -> replay

The ordering is the property under test. A worksheet is created only after the
references it will cite exist; one written first and back-filled would briefly
name artifacts that were not there, and "briefly" is exactly when a crash
happens.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import json
import re

import pytest

from src.workspace.generate import generate, new_worksheet_id, run_id_for
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import from_json

OWNER = "pilot"
#: SPY, not VTI: the committed price fixture holds SPY and not VTI, and a
#: journey test whose assets are absent from the data measures the data gap
#: rather than the journey.
COMPLETE = ("I buy $2000 of SPY on the first trading day of every month in my "
            "taxable brokerage account, reinvest the dividends, and never sell.")


def text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes
    from src.ledger import Ledger

    ledger = Ledger(tmp_path / "public.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
    store = WorkspaceStore(tmp_path / "workspace.db")
    monkeypatch.setattr(workspace_routes, "_store", lambda: store)
    monkeypatch.setattr(workspace_routes, "_parser_client", lambda: None)
    monkeypatch.setattr(workspace_routes, "PRICES",
                        __import__("pathlib").Path(
                            "tests/fixtures/prices_synthetic.parquet"))
    api._bootstrap()
    return TestClient(api.app), store


@pytest.fixture
def compiled():
    from src.mission.compiler import compile_scenario

    return compile_scenario(COMPLETE, name="plan-1", version=1,
                            benchmark_rule="benchmark-policy/public-default@1")


class TestTheJourney:

    def test_saving_a_plan_produces_a_worksheet(self, client):
        api_client, store = client
        from src.mission.compiler import parse

        response = api_client.post(
            "/workspace/save",
            params={"describe": COMPLETE, "plan_id": "plan-1",
                    "confirm_all": "yes"},
            data={"parse": json.dumps(parse(COMPLETE).to_json())},
            follow_redirects=False)
        assert response.status_code == 303

        worksheet = store.worksheet_for_scenario("plan-1", OWNER)
        assert worksheet is not None
        assert worksheet["revision"] == 1

    def test_the_worksheet_cites_a_run_that_exists(self, client):
        api_client, store = client
        from src.mission.compiler import parse

        api_client.post("/workspace/save",
                        params={"describe": COMPLETE, "plan_id": "plan-1",
                                "confirm_all": "yes"},
                        data={"parse": json.dumps(parse(COMPLETE).to_json())},
                        follow_redirects=False)

        record = store.worksheet_for_scenario("plan-1", OWNER)
        worksheet = from_json(record["payload"])
        assert worksheet.primary_run_ref
        assert store.get_run(worksheet.primary_run_ref, OWNER) is not None, (
            "a worksheet must not name a run that does not exist")

    def test_the_saved_worksheet_opens(self, client):
        api_client, _store = client
        from src.mission.compiler import parse

        api_client.post("/workspace/save",
                        params={"describe": COMPLETE, "plan_id": "plan-1",
                                "confirm_all": "yes"},
                        data={"parse": json.dumps(parse(COMPLETE).to_json())},
                        follow_redirects=False)

        identifier = _store.worksheet_for_scenario("plan-1", OWNER)["worksheet_id"]
        page = api_client.get(f"/workspace/research/{identifier}")
        assert page.status_code == 200
        assert "SPY" in text(page.text)


class TestGenerationIsDeterministic:

    def test_a_run_id_names_what_produced_it(self, compiled):
        first = run_id_for("plan-1", compiled.scenario.content_hash, "2026-08-01")
        second = run_id_for("plan-1", compiled.scenario.content_hash, "2026-08-01")
        assert first == second
        assert first.startswith("run-plan-1-")

    def test_a_different_scenario_gets_a_different_run_id(self, compiled):
        assert run_id_for("plan-1", compiled.scenario.content_hash, "t") != \
            run_id_for("plan-1", "a-different-hash", "t")

    def test_no_result_payload_is_copied_onto_the_worksheet(self, tmp_path,
                                                            compiled):
        """The worksheet names the run; the run owns the figures."""
        store = WorkspaceStore(tmp_path / "w.db")
        store.save_plan(plan_id="plan-1", owner=OWNER, scenario=compiled.scenario,
                        stated_text=COMPLETE, saved_at="2026-08-01T00:00:00Z")
        worksheet = generate(
            store, plan_id="plan-1", owner=OWNER, scenario=compiled.scenario,
            run={"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": []}, "final_value": 12345.0},
            comparison={}, ran_at="2026-08-01T00:00:00Z")

        blob = json.dumps(worksheet.to_json())
        assert "12345" not in blob


class TestReSavingMakesARevision:

    def test_a_new_run_produces_a_new_revision(self, tmp_path, compiled):
        """The figures someone already read stay readable."""
        store = WorkspaceStore(tmp_path / "w.db")
        store.save_plan(plan_id="plan-1", owner=OWNER, scenario=compiled.scenario,
                        stated_text=COMPLETE, saved_at="2026-08-01T00:00:00Z")
        first = generate(store, plan_id="plan-1", owner=OWNER,
                         scenario=compiled.scenario,
                         run={"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}}, comparison={},
                         ran_at="2026-08-01T00:00:00Z")
        second = generate(store, plan_id="plan-1", owner=OWNER,
                          scenario=compiled.scenario,
                          run={"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}}, comparison={},
                          ran_at="2027-01-01T00:00:00Z")

        assert first.revision == 1 and second.revision == 2
        assert second.parent_revision == 1
        assert second.change_reason
        assert store.get_worksheet(first.worksheet_id, OWNER, 1) is not None

    def test_an_identical_rerun_makes_no_revision(self, tmp_path, compiled):
        """A revision per page view would bury the changes that matter."""
        store = WorkspaceStore(tmp_path / "w.db")
        store.save_plan(plan_id="plan-1", owner=OWNER, scenario=compiled.scenario,
                        stated_text=COMPLETE, saved_at="2026-08-01T00:00:00Z")
        for _ in range(3):
            generate(store, plan_id="plan-1", owner=OWNER,
                     scenario=compiled.scenario, run={"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}},
                     comparison={}, ran_at="2026-08-01T00:00:00Z")
        assert len(store.worksheet_revisions(store.worksheet_for_scenario("plan-1", OWNER)["worksheet_id"],
                                             OWNER)) == 1


class TestNothingIsCitedBeforeItExists:

    def test_no_run_means_no_worksheet(self, tmp_path, compiled):
        """A worksheet whose performance block can never be filled looks like a
        result that has not loaded."""
        store = WorkspaceStore(tmp_path / "w.db")
        assert generate(store, plan_id="plan-1", owner=OWNER,
                        scenario=compiled.scenario, run={}, comparison={},
                        ran_at="2026-08-01T00:00:00Z") is None
        assert store.worksheet_for_scenario("plan-1", OWNER) is None

    def test_the_run_is_recorded_before_the_worksheet(self, tmp_path, compiled):
        """Ordering asserted through the store, not by reading the source."""
        store = WorkspaceStore(tmp_path / "w.db")
        store.save_plan(plan_id="plan-1", owner=OWNER, scenario=compiled.scenario,
                        stated_text=COMPLETE, saved_at="2026-08-01T00:00:00Z")

        recorded = []
        original = store.record_run

        def watched(**kwargs):
            recorded.append(("run", kwargs["run_id"]))
            return original(**kwargs)

        store.record_run = watched
        saved = store.save_worksheet

        def watched_worksheet(worksheet):
            recorded.append(("worksheet", worksheet.worksheet_id))
            return saved(worksheet)

        store.save_worksheet = watched_worksheet
        generate(store, plan_id="plan-1", owner=OWNER,
                 scenario=compiled.scenario, run={"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}},
                 comparison={}, ran_at="2026-08-01T00:00:00Z")

        assert [kind for kind, _ in recorded] == ["run", "worksheet"]
