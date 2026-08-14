"""The three routes, at the HTTP layer.

`test_login_flow.py` covers the protocol. This covers what the routes decide:
that a deployment with no provider says so rather than 404ing, that the login
link cannot be turned into a redirect off this site, and that a callback with
nothing behind it never sets a cookie.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.deploy.context import IdentityTarget
from src.workspace.auth_routes import SESSION_COOKIE


@pytest.fixture
def client():
    from src.api import app

    return TestClient(app, follow_redirects=False)


def configure(monkeypatch, **fields):
    """Point the routes at a declared provider without touching the process's
    real configuration."""
    target = IdentityTarget(**fields)
    monkeypatch.setattr("src.workspace.auth_routes._target", lambda: target)
    return target


class TestADeploymentWithNoProvider:
    """The pilot's own configuration, which has to keep answering."""

    def test_login_says_so_rather_than_404(self, client, monkeypatch):
        configure(monkeypatch)
        response = client.get("/auth/login")
        assert response.status_code == 503
        assert "no identity provider" in response.text.lower()

    def test_the_callback_says_so_too(self, client, monkeypatch):
        configure(monkeypatch)
        assert client.get("/auth/callback?code=x&state=y").status_code == 503

    def test_neither_sets_a_session(self, client, monkeypatch):
        configure(monkeypatch)
        assert SESSION_COOKIE not in client.get("/auth/login").cookies


class TestTheLoginLinkCannotLeaveTheSite:
    """`?next=` is attacker-supplied and ends up in a redirect.

    An open redirect here is worth having: the link starts on quantify.club,
    carries its name, and lands wherever it was told — which is exactly the
    shape of a credible phishing link.
    """

    @pytest.fixture(autouse=True)
    def _provider(self, monkeypatch):
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1",
                  public_base_url="https://quantify.test")
        monkeypatch.setattr(
            "src.deploy.login.discovery",
            lambda issuer, internal="", timeout=10.0: {
                "authorization_endpoint": f"{issuer}/authorize"})

    @pytest.mark.parametrize("destination", [
        "https://elsewhere.test/steal",
        "//elsewhere.test/steal",
        "http://elsewhere.test",
    ])
    def test_an_absolute_destination_is_discarded(self, client, destination):
        import urllib.parse

        from src.deploy.login import Flow

        response = client.get("/auth/login", params={"next": destination})
        assert response.status_code == 303
        # The destination is carried in the flow cookie, which is what the
        # callback later redirects to — so that is where to look for it.
        raw = response.cookies.get("quantify_login")
        assert raw, "no login cookie was set"
        assert Flow.from_cookie(urllib.parse.unquote(raw)).destination \
            == "/workspace"

    def test_a_path_on_this_site_is_kept(self, client):
        import urllib.parse

        from src.deploy.login import Flow

        response = client.get("/auth/login", params={"next": "/workspace/new"})
        raw = urllib.parse.unquote(response.cookies.get("quantify_login"))
        assert Flow.from_cookie(raw).destination == "/workspace/new"

    def test_the_redirect_goes_to_the_provider(self, client):
        response = client.get("/auth/login")
        assert response.headers["location"].startswith(
            "https://auth.example.test/authorize?")


class TestACallbackWithNothingBehindIt:
    @pytest.fixture(autouse=True)
    def _provider(self, monkeypatch):
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1",
                  public_base_url="https://quantify.test")

    def test_no_login_cookie_is_refused(self, client):
        response = client.get("/auth/callback?code=x&state=y")
        assert response.status_code == 400
        assert SESSION_COOKIE not in response.cookies

    def test_a_provider_error_is_shown_in_its_own_words(self, client):
        response = client.get(
            "/auth/callback?error=access_denied"
            "&error_description=The+person+pressed+cancel")
        assert "pressed cancel" in response.text.lower()
        assert SESSION_COOKIE not in response.cookies


class TestSigningOut:
    def test_it_clears_the_session(self, client, monkeypatch):
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        response = client.get("/auth/logout")
        assert response.status_code == 303
        # An expiry in the past is how a cookie is deleted; the header has to
        # carry it, or the browser keeps the session.
        assert SESSION_COOKIE in response.headers.get("set-cookie", "")
