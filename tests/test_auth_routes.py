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


class TestSigningInIsReachableFromAPage:
    """A route nobody links to is a feature nobody has.

    `/auth/login` existed, was tested, and was deployed — and no page mentioned
    it. From a browser the deployment looked exactly like one with no login at
    all, which is what it was reported as. The flow was never the defect;
    reachability was.
    """

    def rendered(self, monkeypatch, **fields):
        from src.workspace.routes import TEMPLATES, _identity_state

        configure(monkeypatch, **fields)
        monkeypatch.setitem(TEMPLATES.env.globals, "identity_state",
                            lambda: _identity_state())
        return TEMPLATES.env.get_template("base.html").render(
            data_notice=lambda: None)

    def test_a_deployment_with_a_provider_offers_a_link(self, monkeypatch):
        page = self.rendered(monkeypatch, issuer="https://auth.example.test",
                             audience="client-1", client_id="client-1")
        assert "/auth/login" in page
        assert "Sign in" in page

    def test_a_deployment_without_one_offers_nothing(self, monkeypatch):
        """The pilot's own configuration. A sign-in link on a build with no
        provider is a door onto a wall."""
        page = self.rendered(monkeypatch)
        assert "/auth/login" not in page

    def test_a_signed_in_viewer_is_named_and_offered_the_way_out(
            self, monkeypatch):
        from src.deploy.identity import Identity
        from src.workspace.routes import _LOOKING

        token = _LOOKING.set(Identity(subject="user-1",
                                      email="someone@example.test"))
        try:
            page = self.rendered(monkeypatch,
                                 issuer="https://auth.example.test",
                                 audience="client-1", client_id="client-1")
        finally:
            _LOOKING.reset(token)

        assert "someone@example.test" in page
        assert "/auth/logout" in page
        assert "/auth/login" not in page


class TestTheWorkspaceRequiresASession:
    """The change that makes signing in mean something.

    Before this, `/auth/login` existed, worked, and governed nothing: the
    workspace was guarded by a shared basic-auth password at the proxy, and a
    signed-in visitor saw exactly what a visitor with the password saw. The
    login was decoration, which is indistinguishable from no login at all —
    and was reported as exactly that.
    """

    @pytest.fixture
    def client(self):
        from src.api import app

        return TestClient(app, follow_redirects=False)

    def test_a_visitor_without_a_session_is_sent_to_sign_in(self, client,
                                                            monkeypatch):
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        response = client.get("/workspace/")
        assert response.status_code == 303
        assert response.headers["location"].startswith("/auth/login?next=")

    def test_the_page_they_asked_for_is_carried_through(self, client,
                                                        monkeypatch):
        """Landing at the front door after signing in loses the thing they
        were trying to reach."""
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        response = client.get("/workspace/new")
        assert "next=/workspace/new" in response.headers["location"]

    def test_signing_in_is_not_itself_behind_the_gate(self, client,
                                                      monkeypatch):
        """A login that required a login could not be started."""
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        assert client.get("/auth/login").status_code in (303, 502, 503)

    def test_the_public_surfaces_stay_public(self, client, monkeypatch):
        """The research library and the dashboard are published on purpose;
        putting them behind accounts would be a different product."""
        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        for path in ("/health/live", "/info"):
            assert client.get(path).status_code == 200, path

    def test_a_deployment_without_accounts_is_not_locked_out(self, client,
                                                             monkeypatch):
        """The configuration the pilot ran under for months. With no issuer
        there is nothing to sign in to, so requiring a session would refuse
        everybody — the proxy's shared password is the guard there, and this
        middleware must stand aside."""
        configure(monkeypatch)
        response = client.get("/workspace/")
        assert response.status_code != 303 or "auth/login" not in \
            response.headers.get("location", "")

    def test_a_verified_viewer_is_let_through(self, client, monkeypatch):
        from src.deploy.identity import Identity

        configure(monkeypatch, issuer="https://auth.example.test",
                  audience="client-1", client_id="client-1")
        monkeypatch.setattr("src.workspace.auth_routes.signed_in",
                            lambda request: Identity(subject="user-1"))
        response = client.get("/workspace/")
        assert response.status_code != 303 or "auth/login" not in \
            response.headers.get("location", "")
