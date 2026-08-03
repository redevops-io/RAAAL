"""The three-lane panel, served from persisted records.

The route resolves and arranges. It matches nothing, computes no dates and
decides no statuses — those were decided when the reconciliation was derived,
and deciding them again would produce a second answer that disagrees on exactly
the rows that are hard to call.
"""
from __future__ import annotations

import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import src.workspace.routes as routes
from src.api import app
from src.mission.rsu_reconcile import (
    MATCHING_POLICY_VERSION,
    ObservedEvent,
    PlannedEvent,
    reconcile,
)
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

OWNER = "pilot"
AS_OF = "2026-07-10"

JUNE = PlannedEvent(event_id="plan-jun", grant_ref="grant/g1",
                    expected_date="2026-06-15", employer_asset="ACME",
                    expected_gross_shares="100.0", expected_withheld_shares="22.0",
                    expected_delivered_shares="78.0")
SEPT = replace(JUNE, event_id="plan-sep", expected_date="2026-09-15")


def observation(**overrides) -> ObservedEvent:
    base = dict(observation_id="obs-1", observed_date="2026-07-02",
                effective_date="2026-06-19", grant_ref="grant/g1",
                employer_asset="ACME", gross_shares="100.0",
                withheld_shares="22.0", delivered_shares="78.0",
                evidence_ref="statement/june")
    base.update(overrides)
    return ObservedEvent(**base)


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    """A worksheet with a late June vest and a September vest not yet due."""
    store = WorkspaceStore(tmp_path / "w.db")
    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_now", lambda: f"{AS_OF}T00:00:00Z")
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="plan-1", primary_run_ref="run-0",
                                created_at="t0"))

    def load(planned, observed):
        for one in planned:
            store.record_planned_event(
                owner=OWNER, worksheet_id="ws-1", event=one, plan_revision=1,
                created_at="t0",
                matching_policy_version=MATCHING_POLICY_VERSION)
        for one in observed:
            store.record_observed_event(owner=OWNER, worksheet_id="ws-1",
                                        event=one, created_at="t1")
        for row in reconcile(list(planned), list(observed), as_of=AS_OF):
            store.record_reconciliation(owner=OWNER, worksheet_id="ws-1",
                                        reconciliation=row)

    load([JUNE, SEPT], [observation()])
    return store, load


@pytest.fixture
def client():
    return TestClient(app)


def body(client) -> str:
    response = client.get("/workspace/research/ws-1/tracking")
    assert response.status_code == 200
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text))


class TestTheRouteRendersFromRecords:

    def test_it_serves_the_panel(self, tracked, client):
        assert "Planned against observed" in body(client)

    def test_a_missing_worksheet_is_404(self, tracked, client):
        assert client.get(
            "/workspace/research/nope/tracking").status_code == 404

    def test_it_shows_both_planned_events(self, tracked, client):
        page = body(client)
        assert "2026-06-15" in page
        assert "2026-09-15" in page


class TestItRendersWithTheReconcilerBroken:
    """The guard. Verification is best-effort; history is not."""

    def test_the_page_still_returns(self, tracked, client, monkeypatch):
        import src.mission.rsu_reconcile as engine

        def explode(*args, **kwargs):
            raise AssertionError("the route re-reconciled")

        monkeypatch.setattr(engine, "reconcile", explode)
        monkeypatch.setattr(engine, "_could_match", explode)
        monkeypatch.setattr(engine, "_variances", explode)
        monkeypatch.setattr(engine, "_days_apart", explode)

        assert client.get(
            "/workspace/research/ws-1/tracking").status_code == 200

    def test_the_rows_still_equal_the_persisted_records(self, tracked, client,
                                                        monkeypatch):
        store, _ = tracked
        import src.mission.rsu_reconcile as engine

        stored = {one["reconciliation_id"]: one["status"]
                  for one in store.reconciliations("ws-1", OWNER)}
        monkeypatch.setattr(engine, "reconcile",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("re-reconciled")))

        page = body(client)
        assert "Late" in page
        assert "Not yet due" in page
        assert set(stored.values()) == {"LATE", "PENDING"}

    def test_verification_degrades_rather_than_failing_the_page(
            self, tracked, client, monkeypatch):
        """A record the user wrote is not less real because this build cannot
        re-judge it."""
        import src.mission.rsu_reconcile as engine

        monkeypatch.setattr(engine, "reconcile",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
        page = body(client)
        assert "could not be verified" in page

    def test_verification_succeeds_when_the_engine_works(self, tracked, client):
        assert "could not be verified" not in body(client)


class TestTheDisplayDistinctions:

    def test_a_future_event_reads_as_pending_not_missing(self, tracked, client):
        page = body(client)
        assert "Not yet due" in page
        assert "Missing" not in page

    def test_an_overdue_event_is_not_shown_as_absent(self, tmp_path,
                                                     monkeypatch, client):
        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        monkeypatch.setattr(routes, "_now", lambda: "2026-07-10T00:00:00Z")
        store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                    scenario_ref="p", primary_run_ref="r",
                                    created_at="t0"))
        store.record_planned_event(
            owner=OWNER, worksheet_id="ws-1", event=JUNE, plan_revision=1,
            created_at="t0", matching_policy_version=MATCHING_POLICY_VERSION)
        for row in reconcile([JUNE], [], as_of="2026-07-10"):
            store.record_reconciliation(owner=OWNER, worksheet_id="ws-1",
                                        reconciliation=row)

        page = body(client)
        assert "Overdue" in page
        assert "Confirmed not received" not in page

    def test_a_late_effective_date_reads_as_late(self, tracked, client):
        assert "Late" in body(client)

    def test_a_late_report_with_an_on_time_date_reads_as_matched(
            self, tmp_path, monkeypatch, client):
        """Reported 2 July, settled 15 June. On time."""
        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        monkeypatch.setattr(routes, "_now", lambda: "2026-07-10T00:00:00Z")
        store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                    scenario_ref="p", primary_run_ref="r",
                                    created_at="t0"))
        punctual = observation(effective_date="2026-06-15",
                               observed_date="2026-07-02")
        store.record_planned_event(
            owner=OWNER, worksheet_id="ws-1", event=JUNE, plan_revision=1,
            created_at="t0", matching_policy_version=MATCHING_POLICY_VERSION)
        store.record_observed_event(owner=OWNER, worksheet_id="ws-1",
                                    event=punctual, created_at="t1")
        for row in reconcile([JUNE], [punctual], as_of="2026-07-10"):
            store.record_reconciliation(owner=OWNER, worksheet_id="ws-1",
                                        reconciliation=row)

        page = body(client)
        assert "As planned" in page
        assert "Late" not in page

    def test_both_dates_are_shown_for_an_observation(self, tracked, client):
        page = body(client)
        assert "effective 2026-06-19" in page
        assert "reported 2026-07-02" in page

    def test_an_ambiguous_row_chooses_nothing(self, tmp_path, monkeypatch,
                                              client):
        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        monkeypatch.setattr(routes, "_now", lambda: "2026-07-10T00:00:00Z")
        store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                    scenario_ref="p", primary_run_ref="r",
                                    created_at="t0"))
        near = replace(JUNE, event_id="plan-jun-b", expected_date="2026-06-17")
        one = observation(effective_date="2026-06-16")
        for planned in (JUNE, near):
            store.record_planned_event(
                owner=OWNER, worksheet_id="ws-1", event=planned,
                plan_revision=1, created_at="t0",
                matching_policy_version=MATCHING_POLICY_VERSION)
        store.record_observed_event(owner=OWNER, worksheet_id="ws-1",
                                    event=one, created_at="t1")
        for row in reconcile([JUNE, near], [one], as_of="2026-07-10"):
            store.record_reconciliation(owner=OWNER, worksheet_id="ws-1",
                                        reconciliation=row)

        page = body(client)
        assert "Could match more than one plan" in page
        assert "As planned" not in page

    def test_a_correction_leaves_both_observations_visible(self, tracked,
                                                           client):
        store, _ = tracked
        store.record_observed_event(
            owner=OWNER, worksheet_id="ws-1",
            event=observation(observation_id="obs-2", delivered_shares="70.0"),
            created_at="t2", supersedes="obs-1")

        ids = {one["observed_event_id"]
               for one in store.observed_events("ws-1", OWNER)}
        assert ids == {"obs-1", "obs-2"}
        assert body(client)


class TestCounterfactualsStayOutOfTheLanes:

    def test_they_render_in_their_own_section(self, tracked):
        from jinja2 import Environment, FileSystemLoader

        from src.workspace.reconciliation_view import RSUReconciliationView

        store, _ = tracked
        view = RSUReconciliationView.from_records(
            store.planned_events("ws-1", OWNER),
            store.observed_events("ws-1", OWNER),
            store.reconciliations("ws-1", OWNER),
            counterfactuals=({"counterfactual_id": "cf-1",
                              "label": "HYPOTHETICAL — this did not happen",
                              "changed_dimension": "blackout_timing",
                              "isolates": "blackout timing"},))

        env = Environment(loader=FileSystemLoader("src/workspace/templates"))
        html = env.get_template("_rsu_reconciliation.html").render(
            view=view.to_json())
        flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))

        assert "Hypothetical" in flat
        assert "did not happen" in flat
        assert "None of them is a record of what occurred" in flat

    def test_the_lane_table_does_not_contain_them(self, tracked):
        from jinja2 import Environment, FileSystemLoader

        from src.workspace.reconciliation_view import RSUReconciliationView

        store, _ = tracked
        view = RSUReconciliationView.from_records(
            store.planned_events("ws-1", OWNER),
            store.observed_events("ws-1", OWNER),
            store.reconciliations("ws-1", OWNER),
            counterfactuals=({"counterfactual_id": "cf-1",
                              "label": "HYPOTHETICAL"},))

        env = Environment(loader=FileSystemLoader("src/workspace/templates"))
        html = env.get_template("_rsu_reconciliation.html").render(
            view=view.to_json())
        table = html[html.index("<table"):html.index("</table>")]
        assert "cf-1" not in table


class TestTheTemplateDecidesNothing:

    def source(self) -> str:
        """Comments stripped, as Jinja strips them.

        The comments explain that the wording comes from the view model, and
        say the wording to do so. Matching them is the same mistake as matching
        "vest" inside "uninvested".
        """
        from tests.template_source import emitted

        return emitted("_rsu_reconciliation.html")

    def test_it_contains_no_date_or_status_logic(self):
        body = self.source()
        for pattern in (r"\{\%[^%]*\bif\b[^%]*\bdate\b[^%]*[<>]",
                        r"\{\%[^%]*\bstatus\b\s*==\s*'(?!.*lower)",
                        r"\{\{[^}]*[-+*/]\s*\d"):
            assert not re.search(pattern, body), pattern

    def test_the_verdict_wording_comes_from_the_view_model(self):
        """So "Not yet due" cannot become "Missing" by a template edit."""
        body = self.source()
        assert "row.verdict" in body
        for wording in ("Not yet due", "Overdue", "Confirmed not received"):
            assert wording not in body
