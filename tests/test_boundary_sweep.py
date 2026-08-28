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
    BoundaryClass,
    Exposure,
    UndeclaredEndpoint,
    boundary_for,
    login_required,
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


class TestEveryRouteHasOneOfTheFourClasses:
    """The finer taxonomy (§5): every live route is exactly one of the four
    plan-named classes or INFRASTRUCTURE, and its coarse `exposure` agrees with
    the class it was given."""

    FOUR_PLUS_INFRA = {
        BoundaryClass.PUBLIC_RESEARCH,
        BoundaryClass.PUBLIC_EVALUATION,
        BoundaryClass.AUTHENTICATED_PERSISTENCE,
        BoundaryClass.PRIVATE_FINANCIAL_STATE,
        BoundaryClass.INFRASTRUCTURE,
    }

    def test_every_live_route_maps_to_exactly_one_class(self, app_paths):
        for path in app_paths:
            entry = boundary_for(path)
            assert entry.boundary_class in self.FOUR_PLUS_INFRA, (
                f"{path} has class {entry.boundary_class}, not one of the four "
                "plan-named classes or INFRASTRUCTURE")

    def test_the_coarse_exposure_agrees_with_the_class(self, app_paths):
        """A public class may never resolve to PRIVATE exposure, nor the
        reverse — except the evaluation surfaces mounted under a private store,
        which keep PRIVATE artifact lineage on purpose while their access is
        public. That single, declared exception is the only place the two axes
        differ."""
        for path in app_paths:
            entry = boundary_for(path)
            cls = entry.boundary_class
            if cls.requires_login:
                assert entry.exposure is Exposure.PRIVATE, path
            elif cls is BoundaryClass.INFRASTRUCTURE:
                assert entry.exposure is Exposure.INFRASTRUCTURE, path
            elif cls is BoundaryClass.PUBLIC_RESEARCH:
                assert entry.exposure is Exposure.PUBLIC, path
            else:  # PUBLIC_EVALUATION
                assert entry.exposure in (Exposure.PUBLIC, Exposure.PRIVATE), path

    def test_everything_under_a_private_mount_has_private_lineage(self, app_paths):
        """The invariant that lets the evaluation carve-outs be public-access
        without leaking: whatever is served under `/workspace` or `/pilot` — the
        try-it entries included — carries PRIVATE artifact lineage."""
        for path in app_paths:
            if path.startswith("/workspace") or path.startswith("/pilot"):
                assert boundary_for(path).exposure is Exposure.PRIVATE, (
                    f"{path} is under a private mount but its exposure is not "
                    "PRIVATE")


class TestTheMiddlewareDecisionIsTheManifests:
    """The property §5 asks for: the login decision the auth middleware makes
    cannot diverge from the manifest, because it is *computed from* the manifest.

    Proven two ways: the derived decision reproduces the previous hand-written
    string-list logic exactly for every live path (so nothing regressed), and
    the constants the middleware and the deploy acceptance check still import are
    themselves derived from the manifest.
    """

    #: The logic as it was before consolidation, frozen here as an independent
    #: oracle. If the manifest-derived decision ever stops matching this for a
    #: real route, the refactor changed behaviour and this fails.
    OLD_PRIVATE_PREFIXES = ("/workspace", "/pilot")
    OLD_PUBLIC_WITHIN_PRIVATE = ("/workspace/new", "/pilot/answer")

    def _old_decision(self, path: str) -> bool:
        private = any(path.startswith(p) for p in self.OLD_PRIVATE_PREFIXES)
        public = any(path == p or path.startswith(p + "/")
                     for p in self.OLD_PUBLIC_WITHIN_PRIVATE)
        return private and not public

    def test_the_derived_decision_matches_the_old_logic_for_every_path(
            self, app_paths):
        disagreements = [
            (p, self._old_decision(p), login_required(p))
            for p in app_paths
            if self._old_decision(p) != login_required(p)
        ]
        assert disagreements == [], (
            "the manifest-derived login decision no longer matches the "
            f"pre-consolidation behaviour: {disagreements}")

    def test_login_required_agrees_with_the_class_for_every_path(self, app_paths):
        for path in app_paths:
            assert login_required(path) is boundary_for(path).requires_login

    def test_the_middleware_constants_are_derived_from_the_manifest(self):
        import src.api as api
        from src.workspace.boundary_manifest import (gated_prefixes,
                                                     public_within_gated)

        assert tuple(api.PRIVATE_PREFIXES) == tuple(gated_prefixes())
        assert tuple(api.PUBLIC_WITHIN_PRIVATE) == tuple(public_within_gated())
        # The gated mounts are still exactly the two workspace routers, so the
        # deploy acceptance check probes each once as it always has.
        assert set(api.PRIVATE_PREFIXES) == {"/pilot", "/workspace"}


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
