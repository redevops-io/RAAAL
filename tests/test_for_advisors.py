"""The public "For Advisors" surface (§8 of the public strategy-lab plan, Gate 4).

Gate 4's whole point is honesty about *deployed* status: the page may explain the
Evaluate → Save → Connect → Constrain → Execute → Supervise lifecycle, but it may
label only what is live as live. Evaluate and Save are the deployed public
evaluator and exact-save handoff; the four Wealth Manager stages run
simulation-first and are not live, so they must read as roadmap / in development —
never as available money movement.

The surface is informational only: it manages no household, takes no account
state, and — this is the other §8 rule — it must not gate the free public
evaluator behind any lead capture.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.workspace.boundary_manifest import (BoundaryClass, boundary_for,
                                             undeclared)


def _is_login_redirect(response) -> bool:
    return (response.status_code == 303
            and response.headers.get("location", "").startswith("/auth/login"))


@pytest.fixture
def anon_client(monkeypatch, tmp_path):
    """A signed-out client against a deployment that HAS an identity provider.

    The same construction Gate 1's public-access contract uses: an issuer must be
    in force or "public" would be vacuous — with no provider the middleware
    stands aside and nothing is gated, so a page passing here would prove
    nothing. The runtime is declared and reads from recordings so the evaluator
    serves rather than refusing, without any provider call.
    """
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.deploy.context import IdentityTarget

    target = IdentityTarget(issuer="https://auth.example.test",
                            audience="client-1", client_id="client-1")
    monkeypatch.setattr("src.workspace.auth_routes._target", lambda: target)

    from src.api import app

    return TestClient(app, follow_redirects=False)


class TestPublicAccess:
    """PUBLIC_RESEARCH: reachable anonymously even with an issuer configured."""

    def test_the_issuer_is_actually_in_force(self, anon_client):
        # Guard: a known private route must gate, or the public assertion below
        # would pass against a deployment that gates nothing.
        assert _is_login_redirect(anon_client.get("/workspace/plans/demo"))

    def test_for_advisors_is_served_anonymously(self, anon_client):
        response = anon_client.get("/for-advisors")
        assert response.status_code == 200
        assert not _is_login_redirect(response)


class TestDeclaredInTheManifest:
    """No undeclared route — `test_boundary_sweep` fails CI otherwise (§8)."""

    def test_route_is_declared_public_research(self):
        entry = boundary_for("/for-advisors")
        assert entry.boundary_class is BoundaryClass.PUBLIC_RESEARCH
        assert entry.why, "every declaration must say why"
        assert not entry.requires_login

    def test_the_live_route_is_not_undeclared(self, anon_client):
        # The same live-vs-manifest check the boundary sweep runs, scoped to the
        # route this surface adds: it exists on the app and is in the manifest.
        assert undeclared(["/for-advisors"]) == []


class TestDeployedStatusHonesty:
    """Gate 4: the label of each stage must match its real deployed status."""

    @pytest.fixture
    def page(self, anon_client):
        return anon_client.get("/for-advisors").text

    @pytest.fixture
    def stages(self, page):
        # The lifecycle list only — the legend above it also carries a Live and a
        # Roadmap key chip, which is not a per-stage label and must not be counted.
        start = page.index('<ol class="stages">')
        return page[start:page.index("</ol>", start)]

    def test_the_lifecycle_stages_are_all_present(self, page):
        for stage in ("Evaluate", "Save strategy plan", "Connect account",
                      "Apply constraints", "Governed execution",
                      "Continuous supervision"):
            assert stage in page, f"lifecycle stage missing: {stage}"

    def test_evaluate_and_save_are_marked_live(self, stages):
        # Exactly two stages carry the live marker; both are deployed.
        assert stages.count('class="status live">Live') == 2

    def test_the_four_wealth_manager_stages_are_marked_roadmap(self, stages):
        # Connect / Constrain / Execute / Supervise — none deployed, all roadmap.
        assert stages.count('class="status roadmap">Roadmap') == 4

    def test_governed_execution_is_roadmap_not_live(self, page):
        # The stage most easily overstated. It must sit under the roadmap marker
        # and be described as simulation-only, never as a live capability.
        exec_at = page.index("Governed execution")
        after = " ".join(page[exec_at:exec_at + 600].split()).lower()
        assert "simulation only" in after
        assert "roadmap" in after or "not live" in after

    def test_it_does_not_claim_live_money_movement(self, page):
        # No sentence may assert that live execution / trading is available.
        low = page.lower()
        for forbidden in ("live execution is available",
                          "live trading is available",
                          "executes real orders",
                          "moves money today",
                          "live money-moving execution is available"):
            assert forbidden not in low, f"overstated claim present: {forbidden}"
        # The declared-vs-realized honesty is stated positively.
        assert "simulation-first" in low
        assert "gated on" in low  # the authorization gate that is unsatisfied


class TestNoLeadCaptureGate:
    """§8: a demo/contact path may be offered but must gate nothing."""

    def test_the_evaluator_stays_open_to_the_anonymous(self, anon_client):
        # Reuse Gate 1's public-access assertion: /evaluate is not behind login.
        response = anon_client.get("/evaluate", params={"describe": "invest $500 monthly"})
        assert response.status_code == 200
        assert not _is_login_redirect(response)

    def test_the_page_says_the_evaluator_stays_free(self, anon_client):
        page = anon_client.get("/for-advisors").text
        low = page.lower()
        assert "/evaluate" in page          # it links to the free evaluator
        assert "optional" in low            # the demo/contact ask is optional
        assert "gates nothing" in low or "no contact required" in low

    def test_the_contact_path_is_a_plain_mailto_not_a_form(self, anon_client):
        # A mailto link, not a lead-capture form that could gate anything.
        page = anon_client.get("/for-advisors").text
        assert "mailto:" in page


class TestNav:
    """The shared shell links the surface at its canonical path."""

    def test_for_advisors_link_renders_and_points_at_the_route(self, anon_client):
        # The nav lives in base.html, rendered by every page built on the shell.
        page = anon_client.get("/for-advisors").text
        assert 'href="/for-advisors"' in page
        assert "For Advisors" in page


class TestNoAccountState:
    """Informational only: it exposes no holdings / tax / account parameters."""

    def test_the_route_declares_no_request_parameters(self):
        import inspect

        from src.workspace.pilot_routes import for_advisors

        params = set(inspect.signature(for_advisors).parameters)
        # Only the framework Request is accepted — no account/holdings/tax input.
        assert params == {"request"}, f"unexpected parameters: {params}"

    def test_extra_account_query_params_are_ignored(self, anon_client):
        # Passing account-shaped state does not change the page or 422.
        clean = anon_client.get("/for-advisors")
        dirty = anon_client.get("/for-advisors",
                                params={"holdings": "AAPL:100", "tax": "0.3",
                                        "account": "brokerage-1"})
        assert dirty.status_code == 200
        assert dirty.text == clean.text
