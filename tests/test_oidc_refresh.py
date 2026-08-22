"""Keeping a session alive with refresh tokens, proved without a network.

`test_identity.py` proves a token is verified correctly and `test_login_flow.py`
proves how one is first obtained. This proves the piece between: what happens
when the ID token expires on schedule. The property under test is that a valid
session is *renewed* rather than silently dropped — and, just as important, that
a session which cannot be renewed is *cleared* rather than left to decay into the
shared `"pilot"` owner.

Nothing here reaches a provider. The discovery document, the key set and the
token endpoint are all answered in-process, so what is exercised is this
application's policy and wiring.
"""
from __future__ import annotations

import base64
import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from src.deploy import login as login_module
from src.deploy.identity import KEYS, LEEWAY_SECONDS, verify
from src.deploy.login import (REFRESH_COOKIE, SESSION_COOKIE, LoginFailed,
                              refresh, refresh_cookie, session_cookie)

ISSUER = "https://auth.example.test"
CLIENT_ID = "client-1"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def _provider(monkeypatch, signing_key):
    """The provider's key set and discovery document, answered in-process.

    `posted` records what a refresh sent to the token endpoint, so the tests can
    assert the grant actually travelled rather than assuming it did. `response`
    is what the endpoint hands back, swappable per test.
    """
    from jwt.api_jwk import PyJWK

    numbers = signing_key.public_key().public_numbers()

    def b64(value: int) -> str:
        length = (value.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(
            value.to_bytes(length, "big")).decode().rstrip("=")

    jwk = PyJWK({"kty": "RSA", "kid": "k1", "use": "sig", "alg": "RS256",
                 "n": b64(numbers.n), "e": b64(numbers.e)})

    class Client:
        def __init__(self, *_a, **_k):
            pass

        def get_signing_key_from_jwt(self, _token):
            return jwk

    def token_for(**overrides) -> str:
        claims = {"sub": "user-9", "iss": ISSUER, "aud": CLIENT_ID,
                  "exp": int(time.time()) + 600, "iat": int(time.time()),
                  "email": "someone@example.test", "name": "Someone"}
        claims.update(overrides)
        return jwt.encode(claims, signing_key, algorithm="RS256",
                          headers={"kid": "k1"})

    posted: Dict[str, Any] = {}
    state = {"response": lambda: {"id_token": token_for(),
                                  "refresh_token": "rotated-refresh"}}

    def post_token(endpoint, fields, *, internal="", timeout=10.0):
        posted["endpoint"] = endpoint
        posted["fields"] = dict(fields)
        return state["response"]()

    discovery_doc = lambda issuer, internal="", timeout=10.0: {
        "jwks_uri": f"{issuer}/keys",
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token"}

    KEYS.clear()
    monkeypatch.setattr("src.deploy.identity.discovery", discovery_doc)
    monkeypatch.setattr("src.deploy.login.discovery", discovery_doc)
    monkeypatch.setattr("jwt.PyJWKClient", Client)
    monkeypatch.setattr(login_module, "_post_token", post_token)
    yield {"token_for": token_for, "posted": posted, "state": state}
    KEYS.clear()


# --- (a) clock-skew leeway on verify --------------------------------------


class TestVerifyToleratesClockSkew:
    def test_a_token_just_past_expiry_within_leeway_is_accepted(self, _provider):
        """A clock a little ahead of the issuer must not read a still-current
        session as expired. The token is past `exp`, but by less than the
        allowance, so it verifies."""
        token = _provider["token_for"](exp=int(time.time()) - (LEEWAY_SECONDS - 5))
        who = verify(token, issuer=ISSUER, audience=CLIENT_ID)
        assert who.subject == "user-9"

    def test_a_token_well_beyond_leeway_is_still_rejected(self, _provider):
        """The allowance is a tolerance, not a reprieve: a token expired by far
        more than the skew window is still refused."""
        token = _provider["token_for"](
            exp=int(time.time()) - (LEEWAY_SECONDS + 600))
        with pytest.raises(jwt.ExpiredSignatureError):
            verify(token, issuer=ISSUER, audience=CLIENT_ID)


# --- (b) the refresh grant -------------------------------------------------


class TestTheRefreshGrant:
    def test_it_posts_a_refresh_token_grant(self, _provider):
        refresh(issuer=ISSUER, client_id=CLIENT_ID, audience=CLIENT_ID,
                refresh_token="the-refresh")
        fields = _provider["posted"]["fields"]
        assert fields["grant_type"] == "refresh_token"
        assert fields["refresh_token"] == "the-refresh"
        assert fields["client_id"] == CLIENT_ID
        assert "openid" in fields["scope"].split()

    def test_it_returns_the_new_id_token_and_the_rotated_refresh(self, _provider):
        who, new_id, new_refresh = refresh(
            issuer=ISSUER, client_id=CLIENT_ID, audience=CLIENT_ID,
            refresh_token="the-refresh")
        assert who.subject == "user-9"
        assert new_id  # a freshly verified ID token
        assert new_refresh == "rotated-refresh"

    def test_it_reuses_the_old_refresh_when_the_response_omits_one(self, _provider):
        """A provider that does not rotate sends no new refresh token; the one
        just used is still good and must be carried forward, or the session
        loses the ability to renew at the next expiry."""
        _provider["state"]["response"] = lambda: {
            "id_token": _provider["token_for"]()}
        _who, _new_id, new_refresh = refresh(
            issuer=ISSUER, client_id=CLIENT_ID, audience=CLIENT_ID,
            refresh_token="still-good")
        assert new_refresh == "still-good"

    def test_a_response_with_no_id_token_is_a_failure(self, _provider):
        _provider["state"]["response"] = lambda: {"access_token": "opaque"}
        with pytest.raises(LoginFailed):
            refresh(issuer=ISSUER, client_id=CLIENT_ID, audience=CLIENT_ID,
                    refresh_token="the-refresh")

    def test_a_refreshed_token_for_another_audience_is_rejected(self, _provider):
        """A renewed token is verified exactly as a fresh one — a refresh does
        not buy a token past the audience check."""
        _provider["state"]["response"] = lambda: {
            "id_token": _provider["token_for"](aud="another-app"),
            "refresh_token": "rotated"}
        with pytest.raises(jwt.PyJWTError):
            refresh(issuer=ISSUER, client_id=CLIENT_ID, audience=CLIENT_ID,
                    refresh_token="the-refresh")


# --- (c, d) transparent refresh inside signed_in ---------------------------


def _configured_target():
    return SimpleNamespace(issuer=ISSUER, audience=CLIENT_ID,
                           client_id=CLIENT_ID, internal_base_url="",
                           configured=True)


def _request(cookies):
    return SimpleNamespace(cookies=cookies, state=SimpleNamespace())


class TestSignedInRefreshesTransparently:
    @pytest.fixture(autouse=True)
    def _wiring(self, monkeypatch):
        from src.workspace import auth_routes

        monkeypatch.setattr(auth_routes, "_target", _configured_target)
        # The session cookie never resolves in these tests; refresh is what
        # decides the outcome.
        monkeypatch.setattr(auth_routes, "viewer", lambda *a, **k: None)

    def test_a_stale_session_with_a_valid_refresh_is_renewed(self, monkeypatch):
        from src.deploy.identity import Identity
        from src.workspace import auth_routes

        # Patch the login module object `auth_routes` actually references, which
        # under pytest's import layout is not necessarily the one reachable as
        # `src.deploy.login`.
        monkeypatch.setattr(
            auth_routes.login_flow, "refresh",
            lambda **_k: (Identity(subject="user-9"), "new-id", "new-refresh"))

        request = _request({SESSION_COOKIE: "stale", REFRESH_COOKIE: "rt"})
        who = auth_routes.signed_in(request)

        assert who is not None and who.subject == "user-9"
        # Stashed for the response middleware to write.
        assert request.state._new_session_token == "new-id"
        assert request.state._new_refresh_token == "new-refresh"
        assert getattr(request.state, "_clear_session", False) is False

    def test_the_outcome_is_cached_and_refresh_runs_at_most_once(self, monkeypatch):
        from src.deploy.identity import Identity
        from src.workspace import auth_routes

        calls = {"n": 0}

        def once(**_k):
            calls["n"] += 1
            return Identity(subject="user-9"), "new-id", "new-refresh"

        monkeypatch.setattr(auth_routes.login_flow, "refresh", once)

        request = _request({SESSION_COOKIE: "stale", REFRESH_COOKIE: "rt"})
        first = auth_routes.signed_in(request)
        second = auth_routes.signed_in(request)

        assert first is second
        assert calls["n"] == 1, "refresh must not run twice in one request"

    def test_a_failed_refresh_marks_the_session_for_clearing(self, monkeypatch):
        from src.workspace import auth_routes

        def boom(**_k):
            raise LoginFailed("the provider revoked it")

        monkeypatch.setattr(auth_routes.login_flow, "refresh", boom)

        request = _request({SESSION_COOKIE: "stale", REFRESH_COOKIE: "rt"})
        who = auth_routes.signed_in(request)

        assert who is None
        assert request.state._clear_session is True
        assert getattr(request.state, "_new_session_token", None) is None

    def test_a_stale_session_with_no_refresh_cookie_is_cleared(self, monkeypatch):
        """A present-but-dead session with nothing to renew it must be cleared,
        not left to read as `pilot` on the next request."""
        from src.workspace import auth_routes

        request = _request({SESSION_COOKIE: "stale"})
        who = auth_routes.signed_in(request)

        assert who is None
        assert request.state._clear_session is True

    def test_a_request_with_no_session_at_all_is_left_alone(self, monkeypatch):
        """Genuinely anonymous: nothing stale to remove, so no deletion is
        ordered."""
        from src.workspace import auth_routes

        request = _request({})
        who = auth_routes.signed_in(request)

        assert who is None
        assert getattr(request.state, "_clear_session", False) is False


# --- (e) the response middleware writes the cookies ------------------------


def _all_set_cookies(response) -> str:
    try:
        return "\n".join(response.headers.get_list("set-cookie"))
    except AttributeError:
        return response.headers.get("set-cookie", "")


class TestTheResponseMiddlewareAppliesTheDecision:
    """Driven through the real application, so the middleware ordering that
    makes `request.state` visible after `call_next` is exercised, not assumed."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api import app

        return TestClient(app, follow_redirects=False)

    @pytest.fixture(autouse=True)
    def _wiring(self, monkeypatch):
        from src.workspace import auth_routes

        monkeypatch.setattr(auth_routes, "_target", _configured_target)
        monkeypatch.setattr(auth_routes, "viewer", lambda *a, **k: None)

    def test_a_renewed_session_is_written_onto_the_response(self, client,
                                                            monkeypatch):
        from src.deploy.identity import Identity
        from src.workspace import auth_routes

        monkeypatch.setattr(
            auth_routes.login_flow, "refresh",
            lambda **_k: (Identity(subject="user-9"), "new-id", "new-refresh"))

        response = client.get(
            "/info", cookies={SESSION_COOKIE: "stale", REFRESH_COOKIE: "rt"})
        header = _all_set_cookies(response)
        assert response.status_code == 200
        assert "quantify_session=new-id" in header
        assert "quantify_refresh=new-refresh" in header

    def test_an_unrenewable_session_has_both_cookies_deleted(self, client,
                                                             monkeypatch):
        from src.workspace import auth_routes

        def boom(**_k):
            raise LoginFailed("revoked")

        monkeypatch.setattr(auth_routes.login_flow, "refresh", boom)

        response = client.get(
            "/info", cookies={SESSION_COOKIE: "stale", REFRESH_COOKIE: "rt"})
        header = _all_set_cookies(response)
        # A deletion is a cookie set to expire immediately; both names appear.
        assert "quantify_session=" in header
        assert "quantify_refresh=" in header
        assert "Max-Age=0" in header or "expires=" in header.lower()


# --- (f) callback sets both cookies; logout deletes both -------------------


class TestCallbackAndLogoutCarryBothCookies:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api import app

        return TestClient(app, follow_redirects=False)

    @pytest.fixture(autouse=True)
    def _configured(self, monkeypatch):
        from src.deploy.context import IdentityTarget
        from src.workspace import auth_routes

        target = IdentityTarget(issuer=ISSUER, audience=CLIENT_ID,
                                client_id=CLIENT_ID,
                                public_base_url="https://quantify.test")
        monkeypatch.setattr(auth_routes, "_target", lambda: target)

    def test_a_successful_callback_sets_the_session_and_refresh_cookies(
            self, client, monkeypatch):
        from src.deploy.identity import Identity
        from src.deploy.login import Flow
        from src.workspace import auth_routes

        flow = Flow(state="s", verifier="v", destination="/workspace")
        monkeypatch.setattr(
            auth_routes, "complete",
            lambda **_k: (Identity(subject="user-9"), "id-tok", "refresh-tok"))

        response = client.get(
            "/auth/callback", params={"code": "c", "state": "s"},
            cookies={"quantify_login": flow.to_cookie()})
        header = _all_set_cookies(response)
        assert response.status_code == 303
        assert "quantify_session=id-tok" in header
        assert "quantify_refresh=refresh-tok" in header

    def test_a_callback_without_a_refresh_token_sets_only_the_session(
            self, client, monkeypatch):
        """A provider that returns no refresh token must not have a blank one
        persisted for it — the session still works, it just cannot renew."""
        from src.deploy.identity import Identity
        from src.deploy.login import Flow
        from src.workspace import auth_routes

        flow = Flow(state="s", verifier="v", destination="/workspace")
        monkeypatch.setattr(
            auth_routes, "complete",
            lambda **_k: (Identity(subject="user-9"), "id-tok", ""))

        response = client.get(
            "/auth/callback", params={"code": "c", "state": "s"},
            cookies={"quantify_login": flow.to_cookie()})
        header = _all_set_cookies(response)
        assert "quantify_session=id-tok" in header
        assert "quantify_refresh=" not in header

    def test_logout_deletes_both_cookies(self, client):
        response = client.get("/auth/logout")
        header = _all_set_cookies(response)
        assert response.status_code == 303
        assert "quantify_session=" in header
        assert "quantify_refresh=" in header


# --- the cookie helper -----------------------------------------------------


class TestTheRefreshCookie:
    def test_it_is_not_readable_by_script_and_not_sent_over_http(self):
        attributes = refresh_cookie("a-refresh-token")
        assert attributes["key"] == REFRESH_COOKIE
        assert attributes["httponly"] is True
        assert attributes["secure"] is True
        assert attributes["samesite"] == "lax"
        assert attributes["path"] == "/"

    def test_it_persists_unlike_the_session_cookie(self):
        """The session cookie expires with its token and carries no max_age;
        the refresh cookie is deliberately kept so a session survives the
        browser closing."""
        assert "max_age" not in session_cookie("a.b.c")
        assert refresh_cookie("rt")["max_age"] > 0
