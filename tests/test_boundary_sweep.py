"""Every endpoint, checked against a declared boundary.

The failure mode is not a route that leaks today. It is a route added next
quarter by someone thinking about a feature rather than about the publisher
position, which inherits whatever the framework defaults to.

So the manifest is checked against the live application: an undeclared route
fails this suite until somebody decides which side it is on. That is the same
rule the artifact model applies everywhere else — a realization with no
declaration is a defect — pointed at the HTTP surface.
"""
from __future__ import annotations

import re

import pytest

from src.mission.boundary import PROHIBITED_KEYS
from src.workspace.boundary_manifest import (
    MANIFEST,
    Exposure,
    UndeclaredEndpoint,
    boundary_for,
    undeclared,
)


def _all_paths(app) -> list:
    """Every reachable path, including those behind an included router.

    `app.routes` returns only the top level: routers added with
    `include_router` appear as a single opaque entry whose sub-paths are not
    exposed. Enumerating it alone silently skipped `/ui` and `/workspace`
    entirely — the two routers this whole manifest exists to police — so the
    check passed while covering neither.

    Walking `routes` recursively catches those, and the OpenAPI schema catches
    anything the walk misses.
    """
    found = set()

    def walk(routes):
        for route in routes:
            path = getattr(route, "path", "")
            if path:
                found.add(path)
            nested = getattr(route, "routes", None) or getattr(
                route, "router", None)
            if nested is not None:
                walk(getattr(nested, "routes", nested))

    walk(app.routes)
    found.update(app.openapi().get("paths", {}))
    return sorted(found)


@pytest.fixture
def app_paths():
    import src.api as api

    return _all_paths(api.app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes
    from src.ledger import Ledger
    from src.workspace.store import WorkspaceStore

    ledger = Ledger(tmp_path / "public.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
    monkeypatch.setattr(workspace_routes, "_store",
                        lambda: WorkspaceStore(tmp_path / "w.db"))
    api._bootstrap()
    return TestClient(api.app)


class TestEveryRouteIsDeclared:
    def test_the_enumeration_reaches_behind_included_routers(self, app_paths):
        """The check is worthless if it cannot see the routers it polices.

        `app.routes` alone returns 26 top-level paths and no `/ui` or
        `/workspace` route at all.
        """
        assert any(p.startswith("/ui") for p in app_paths)
        assert any(p.startswith("/workspace") for p in app_paths)
        assert len(app_paths) > 30

    def test_no_endpoint_is_missing_from_the_manifest(self, app_paths):
        """A new route fails here until its side is decided."""
        missing = undeclared(app_paths)
        assert missing == [], (
            f"undeclared endpoints: {missing}. Every route must state whether it "
            "serves public research or a private workspace."
        )

    def test_the_root_declaration_is_not_a_catch_all(self):
        """Matching '/' as a prefix would make this check pass forever."""
        with pytest.raises(UndeclaredEndpoint):
            boundary_for("/some-future-feature")

    def test_workspace_routes_are_private(self, app_paths):
        for path in app_paths:
            if path.startswith("/workspace"):
                assert boundary_for(path).exposure is Exposure.PRIVATE

    def test_library_routes_are_public(self, app_paths):
        for path in app_paths:
            if path.startswith("/ui"):
                assert boundary_for(path).exposure is Exposure.PUBLIC

    def test_every_declaration_says_why(self):
        for entry in MANIFEST:
            assert entry.why, f"{entry.path} declares a side without a reason"


class TestPublicSurfacesCarryNothingPrivate:
    PUBLIC_PAGES = ["/ui/", "/ui/protocols", "/ui/errata", "/ui/claims",
                    "/ui/findings", "/ui/investigations", "/ui/m/hrp/3"]

    @pytest.mark.parametrize("path", PUBLIC_PAGES)
    def test_no_private_artifact_ids_appear(self, client, path):
        body = client.get(path).text
        for prefix in ("mission/", "intent/", "plan-run/", "scenario/",
                       "proposal/", "observation/"):
            assert prefix not in body, f"{path} exposes a {prefix} artifact"

    @pytest.mark.parametrize("path", PUBLIC_PAGES)
    def test_no_link_into_the_workspace(self, client, path):
        assert "/workspace" not in client.get(path).text

    @pytest.mark.parametrize("path", PUBLIC_PAGES)
    def test_no_personal_field_names_appear(self, client, path):
        body = client.get(path).text.lower()
        for key in ("vesting_schedule", "withholding", "employer",
                    "account_value", "contribution_amount"):
            assert key not in body, f"{path} mentions {key}"

    def test_the_public_api_schema_declares_no_private_paths(self, client):
        schema = client.get("/openapi.json").json()
        private = [p for p in schema.get("paths", {}) if p.startswith("/workspace")]

        assert private, "the workspace is not mounted at all"
        for path in private:
            assert boundary_for(path).exposure is Exposure.PRIVATE


class TestPrivateSurfacesMayCitePublicOnes:
    def test_the_workspace_links_to_the_library(self, client):
        assert "/ui/" in client.get("/workspace/").text

    def test_a_scenario_may_reference_a_published_methodology(self):
        from src.mission.boundary import check_reference

        check_reference("mission/my-plan@1", "methodology/hrp@3")

    def test_the_reverse_is_refused(self):
        from src.mission.boundary import BoundaryViolation, check_reference

        with pytest.raises(BoundaryViolation):
            check_reference("methodology/hrp@3", "mission/my-plan@1")


class TestExtractionCarriesNothingPersonal:
    def test_every_prohibited_key_is_detected_at_depth(self):
        from src.mission.boundary import scan_for_personal_data

        for key in sorted(PROHIBITED_KEYS):
            payload = {"events": [{"when": "x", "then": {key: "value"}}]}
            assert scan_for_personal_data(payload), f"{key} passed the scan"

    def test_a_personal_result_cannot_be_promoted(self):
        """One user's scenario is not a methodology."""
        from src.mission import FlowSchedule, Mission, Objective, Provenance
        from src.mission.boundary import extract_rule

        plan = Mission(name="p", version=1, title="P", objective=Objective.REPLAY,
                       flows=FlowSchedule("monthly", 2000.0),
                       events=[{"trigger": "vest", "employer": "ACME"}],
                       provenance=Provenance())
        extraction = extract_rule(plan)

        assert not extraction.proposable
        assert extraction.report()["public_boundary_check"] == "FAIL"

    def test_a_failed_extraction_is_still_reported(self):
        """It is the clearest evidence of a compiler or schema defect."""
        from src.mission import FlowSchedule, Mission, Objective, Provenance
        from src.mission.boundary import extract_rule

        report = extract_rule(
            Mission(name="p", version=1, title="P", objective=Objective.REPLAY,
                    flows=FlowSchedule("monthly", 2000.0),
                    events=[{"vesting_schedule": "4y1c"}],
                    provenance=Provenance())).report()

        assert report["values_removed"]
        assert report["eligible_for_authoring"] is False


class TestExecutionIsNeverAvailable:
    def test_execution_mode_has_exactly_one_value(self):
        from src.mission import ExecutionMode

        assert [m.value for m in ExecutionMode] == ["NONE"]

    def test_no_endpoint_offers_to_place_an_order(self, client):
        schema = client.get("/openapi.json").json()
        for path in schema.get("paths", {}):
            assert not re.search(r"\b(order|execute|trade|submit)\b", path,
                                 re.IGNORECASE), f"{path} suggests execution"

    def test_a_resolved_proposal_still_reports_nothing_placed(self):
        from src.mission import Proposal, ProposalStatus

        proposal = Proposal(
            proposal_id="p1", plan_id="plan", generated_at="2026-08-17",
            generated_from="disposition", reason="your plan sells at vest")
        accepted = proposal.resolve(ProposalStatus.ACCEPTED)

        assert accepted.to_json()["placed"] is False
        assert accepted.to_json()["execution_mode"] == "NONE"
