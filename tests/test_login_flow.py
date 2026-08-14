"""The login flow, proved against the callbacks that must not create a session.

`test_identity.py` covers token verification. This covers everything around it:
the redirect out, the state comparison, the code exchange, and what the cookie
is worth. As there, the happy path is one test and the rest are attempts.

Nothing here reaches a provider. The discovery document and the token endpoint
are answered in-process, so what is under test is this application's policy.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
from typing import Any, Dict

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from src.deploy import login as flow_module
from src.deploy.identity import KEYS, IdentityUnavailable
from src.deploy.login import (Flow, LoginFailed, begin, complete,
                              session_cookie, viewer)

ISSUER = "https://auth.example.test"
CLIENT_ID = "123456789@quantify"
REDIRECT = "https://quantify.test/auth/callback"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def _provider(monkeypatch, signing_key):
    """The provider, answered in-process.

    Records what the token endpoint was sent, so the tests can assert the
    verifier actually travelled rather than assuming it did.
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

    sent: Dict[str, Any] = {}

    def token_for(**overrides) -> str:
        claims = {"sub": "user-9", "iss": ISSUER, "aud": CLIENT_ID,
                  "exp": int(time.time()) + 600, "iat": int(time.time()),
                  "email": "someone@example.test", "name": "Someone"}
        claims.update(overrides)
        return jwt.encode(claims, signing_key, algorithm="RS256",
                          headers={"kid": "k1"})

    state = {"response": lambda: {"id_token": token_for()}}

    def exchange(*, issuer, client_id, redirect_uri, code, verifier, timeout):
        sent.update({"issuer": issuer, "client_id": client_id,
                     "redirect_uri": redirect_uri, "code": code,
                     "verifier": verifier})
        return state["response"]()

    KEYS.clear()
    monkeypatch.setattr("src.deploy.identity.discovery",
                        lambda issuer, timeout=10.0: {
                            "jwks_uri": f"{issuer}/keys",
                            "authorization_endpoint": f"{issuer}/authorize",
                            "token_endpoint": f"{issuer}/token"})
    monkeypatch.setattr("src.deploy.login.discovery",
                        lambda issuer, timeout=10.0: {
                            "jwks_uri": f"{issuer}/keys",
                            "authorization_endpoint": f"{issuer}/authorize",
                            "token_endpoint": f"{issuer}/token"})
    monkeypatch.setattr("jwt.PyJWKClient", Client)
    monkeypatch.setattr(flow_module, "_exchange", exchange)
    yield {"sent": sent, "state": state, "token_for": token_for}
    KEYS.clear()


def started():
    return begin(issuer=ISSUER, client_id=CLIENT_ID, redirect_uri=REDIRECT)


class TestTheRedirectOut:
    def test_it_carries_a_pkce_challenge_derived_from_the_verifier(self):
        """The challenge must be the hash of the verifier, or the exchange is
        rejected by the provider and PKCE protects nothing."""
        where, flow = started()
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(where).query)

        expected = base64.urlsafe_b64encode(
            hashlib.sha256(flow.verifier.encode()).digest()).decode().rstrip("=")
        assert query["code_challenge"] == [expected]
        assert query["code_challenge_method"] == ["S256"]

    def test_it_asks_for_openid(self):
        """Without this scope the provider is not obliged to return an ID
        token, and there is nothing to establish who signed in."""
        where, _ = started()
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(where).query)
        assert "openid" in query["scope"][0].split()

    def test_the_verifier_never_appears_in_the_redirect(self):
        """It is the secret half. Sending it out with the challenge would make
        the exchange verifiable by whoever saw the redirect."""
        where, flow = started()
        assert flow.verifier not in where

    def test_two_logins_do_not_share_a_state_or_a_verifier(self):
        first, second = started()[1], started()[1]
        assert first.state != second.state
        assert first.verifier != second.verifier

    def test_an_unconfigured_deployment_cannot_start_one(self):
        with pytest.raises(IdentityUnavailable):
            begin(issuer="", client_id=CLIENT_ID, redirect_uri=REDIRECT)
        with pytest.raises(IdentityUnavailable):
            begin(issuer=ISSUER, client_id="", redirect_uri=REDIRECT)


class TestTheCallback:
    def completed(self, flow, **overrides):
        arguments = {"issuer": ISSUER, "client_id": CLIENT_ID,
                     "audience": CLIENT_ID, "redirect_uri": REDIRECT,
                     "code": "the-code", "state": flow.state, "flow": flow}
        arguments.update(overrides)
        return complete(**arguments)

    def test_a_matching_callback_yields_a_verified_identity(self, _provider):
        _, flow = started()
        who, token = self.completed(flow)
        assert who.subject == "user-9"
        assert who.email == "someone@example.test"
        assert token

    def test_the_verifier_is_sent_to_the_token_endpoint(self, _provider):
        """PKCE is only doing anything if the verifier actually travels."""
        _, flow = started()
        self.completed(flow)
        assert _provider["sent"]["verifier"] == flow.verifier

    def test_a_callback_with_the_wrong_state_is_refused(self, _provider):
        """The CSRF check. Without it, an attacker can complete a login they
        started into somebody else's browser."""
        _, flow = started()
        with pytest.raises(LoginFailed):
            self.completed(flow, state="not-the-state")

    def test_a_callback_with_no_state_is_refused(self, _provider):
        _, flow = started()
        with pytest.raises(LoginFailed):
            self.completed(flow, state="")

    def test_the_state_is_checked_before_the_code_is_exchanged(self, _provider):
        """A callback that did not start here should cost nothing — and must
        not hand an attacker-supplied code to the provider at all."""
        _, flow = started()
        _provider["sent"].clear()
        with pytest.raises(LoginFailed):
            self.completed(flow, state="wrong")
        assert not _provider["sent"], (
            "the code was exchanged before the state was compared")

    def test_a_response_with_no_id_token_is_refused(self, _provider):
        """An access token alone says nothing about who signed in."""
        _, flow = started()
        _provider["state"]["response"] = lambda: {"access_token": "opaque"}
        with pytest.raises(LoginFailed):
            self.completed(flow)

    def test_a_token_for_another_audience_never_becomes_a_session(self, _provider):
        """The provider is shared. A token minted for a different application
        is valid, correctly signed, and not evidence about this one."""
        _, flow = started()
        _provider["state"]["response"] = lambda: {
            "id_token": _provider["token_for"](aud="another-app")}
        with pytest.raises(jwt.PyJWTError):
            self.completed(flow)

    def test_an_expired_token_never_becomes_a_session(self, _provider):
        _, flow = started()
        _provider["state"]["response"] = lambda: {
            "id_token": _provider["token_for"](exp=int(time.time()) - 5)}
        with pytest.raises(jwt.ExpiredSignatureError):
            self.completed(flow)


class TestTheFlowCookie:
    def test_it_survives_a_round_trip(self):
        _, flow = started()
        assert Flow.from_cookie(flow.to_cookie()) == flow

    @pytest.mark.parametrize("raw", ["", "not-base64", "e30=", "!!!"])
    def test_a_malformed_one_is_refused_rather_than_crashing(self, raw):
        """It arrives from a browser, so it is attacker-controlled input."""
        with pytest.raises(LoginFailed):
            Flow.from_cookie(raw)


class TestTheSessionCookie:
    def test_it_is_not_readable_by_script_and_not_sent_over_http(self):
        attributes = session_cookie("a.b.c")
        assert attributes["httponly"] is True
        assert attributes["secure"] is True
        assert attributes["samesite"] == "lax"

    def test_a_forged_cookie_is_not_a_viewer(self, _provider, signing_key):
        """The cookie is the token, so the signature is what makes it a
        session. A value somebody typed is not one."""
        assert viewer("not-a-token", issuer=ISSUER, audience=CLIENT_ID) is None

    def test_a_token_signed_by_another_key_is_not_a_viewer(self, _provider):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = jwt.encode({"sub": "somebody-else", "iss": ISSUER,
                             "aud": CLIENT_ID,
                             "exp": int(time.time()) + 600},
                            other, algorithm="RS256", headers={"kid": "k1"})
        assert viewer(forged, issuer=ISSUER, audience=CLIENT_ID) is None

    def test_an_expired_cookie_is_not_a_viewer(self, _provider):
        assert viewer(_provider["token_for"](exp=int(time.time()) - 5),
                      issuer=ISSUER, audience=CLIENT_ID) is None

    def test_a_valid_one_is(self, _provider):
        who = viewer(_provider["token_for"](), issuer=ISSUER,
                     audience=CLIENT_ID)
        assert who is not None and who.subject == "user-9"

    def test_no_cookie_is_not_a_viewer(self, _provider):
        assert viewer(None, issuer=ISSUER, audience=CLIENT_ID) is None
        assert viewer("", issuer=ISSUER, audience=CLIENT_ID) is None

    def test_an_unconfigured_deployment_has_no_viewers(self, _provider):
        """Not a relaxed check: with no audience declared, a token this
        deployment cannot check must not become a session."""
        assert viewer(_provider["token_for"](), issuer=ISSUER,
                      audience="") is None
