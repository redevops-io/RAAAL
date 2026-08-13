"""Token verification, proved against forgeries rather than against happy paths.

A test suite for an authenticator that only checks valid tokens are accepted
proves nothing anybody cares about: the interesting property is that everything
else is rejected. So every test here except the first constructs a token that a
naive implementation would accept — signed by the wrong key, `alg: none`, the
wrong audience, the wrong issuer, expired, or missing a claim — and requires it
to be refused.

The keys are generated in-process and the key set is served from a local
fixture, so nothing here reaches a provider and nothing depends on one being
configured. What is under test is this deployment's policy, not Zitadel.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from src.deploy.identity import (KEYS, Identity, IdentityUnavailable, verify)

ISSUER = "https://auth.example.test"
AUDIENCE = "quantify"


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def keys():
    return {"good": _key(), "attacker": _key()}


@pytest.fixture(autouse=True)
def _local_jwks(monkeypatch, keys):
    """Serve the issuer's key set from memory.

    `PyJWKClient` is replaced rather than the network stubbed, because a test
    that reaches a URL is a test that fails in CI for reasons unrelated to the
    property it asserts.
    """
    from jwt.api_jwk import PyJWK

    public = keys["good"].public_key()
    numbers = public.public_numbers()

    def b64(value: int) -> str:
        import base64
        length = (value.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(
            value.to_bytes(length, "big")).decode().rstrip("=")

    jwk = PyJWK({"kty": "RSA", "kid": "test-key", "use": "sig", "alg": "RS256",
                 "n": b64(numbers.n), "e": b64(numbers.e)})

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_signing_key_from_jwt(self, _token):
            return jwt.PyJWK(jwk._jwk_data) if hasattr(jwk, "_jwk_data") else jwk

    KEYS.clear()
    monkeypatch.setattr("src.deploy.identity.discovery",
                        lambda issuer, timeout=10.0: {
                            "jwks_uri": f"{issuer}/keys"})
    monkeypatch.setattr("jwt.PyJWKClient", Client)
    yield
    KEYS.clear()


def token(keys, *, key: str = "good", algorithm: str = "RS256",
          **overrides: Any) -> str:
    claims: Dict[str, Any] = {
        "sub": "user-1", "iss": ISSUER, "aud": AUDIENCE,
        "exp": int(time.time()) + 600, "iat": int(time.time()),
        "email": "someone@example.test", "name": "Someone",
    }
    claims.update(overrides)
    for absent in [k for k, v in overrides.items() if v is None]:
        claims.pop(absent, None)
    return jwt.encode(claims, keys[key], algorithm=algorithm,
                      headers={"kid": "test-key"})


class TestAValidTokenIsAccepted:
    def test_it_yields_the_subject(self, keys):
        found = verify(token(keys), issuer=ISSUER, audience=AUDIENCE)
        assert isinstance(found, Identity)
        assert found.subject == "user-1"
        assert found.email == "someone@example.test"
        assert found.issuer == ISSUER


class TestEverythingElseIsRefused:
    """Each of these is a token a naive implementation accepts."""

    def test_a_token_signed_by_another_key_is_refused(self, keys):
        """The one that matters most. Anybody can mint a well-formed JWT; the
        signature is the only thing that makes it evidence."""
        with pytest.raises(jwt.PyJWTError):
            verify(token(keys, key="attacker"), issuer=ISSUER,
                   audience=AUDIENCE)

    def test_an_unsigned_token_is_refused(self, keys):
        """`alg: none` is the classic. A verifier that reads the algorithm out
        of the token it is checking is asking the forger which lock to use."""
        claims = {"sub": "user-1", "iss": ISSUER, "aud": AUDIENCE,
                  "exp": int(time.time()) + 600}
        forged = jwt.encode(claims, key="", algorithm="none")
        with pytest.raises(jwt.PyJWTError):
            verify(forged, issuer=ISSUER, audience=AUDIENCE)

    def test_a_token_for_another_audience_is_refused(self, keys):
        """A valid token from the same issuer, minted for a different
        application. Without the audience check, every service trusting this
        provider trusts every other service's tokens."""
        with pytest.raises(jwt.PyJWTError):
            verify(token(keys, aud="some-other-app"), issuer=ISSUER,
                   audience=AUDIENCE)

    def test_a_token_from_another_issuer_is_refused(self, keys):
        with pytest.raises(jwt.PyJWTError):
            verify(token(keys, iss="https://elsewhere.test"), issuer=ISSUER,
                   audience=AUDIENCE)

    def test_an_expired_token_is_refused(self, keys):
        """Expired tokens are perfectly well-formed, which is exactly why the
        check has to be explicit."""
        with pytest.raises(jwt.ExpiredSignatureError):
            verify(token(keys, exp=int(time.time()) - 60), issuer=ISSUER,
                   audience=AUDIENCE)

    @pytest.mark.parametrize("claim", ["sub", "exp", "iss", "aud"])
    def test_a_token_missing_a_required_claim_is_refused(self, keys, claim):
        with pytest.raises(jwt.PyJWTError):
            verify(token(keys, **{claim: None}), issuer=ISSUER,
                   audience=AUDIENCE)


class TestAnUnconfiguredDeploymentHasNoIdentity:
    def test_it_raises_rather_than_accepting(self, keys):
        """Not a relaxed check. A build that cannot name its issuer must not
        verify anything, or a deployment with the variable unset would accept
        tokens from whoever guessed the audience."""
        with pytest.raises(IdentityUnavailable):
            verify(token(keys), issuer="", audience=AUDIENCE)
        with pytest.raises(IdentityUnavailable):
            verify(token(keys), issuer=ISSUER, audience="")

    def test_the_context_reports_it(self):
        from src.deploy.context import resolve

        assert not resolve({}).identity.configured
        assert resolve({"OIDC_ISSUER": ISSUER,
                        "OIDC_AUDIENCE": AUDIENCE}).identity.configured

    def test_no_secret_reaches_the_json(self):
        """`to_json` travels to a page. The client id belongs there — it starts
        a login — and nothing else about the provider does."""
        from src.deploy.context import resolve

        rendered = resolve({"OIDC_ISSUER": ISSUER, "OIDC_AUDIENCE": AUDIENCE,
                            "OIDC_CLIENT_ID": "public-client"}).identity.to_json()
        assert set(rendered) == {"configured", "issuer", "audience",
                                 "client_id"}
        assert "secret" not in json.dumps(rendered).lower()
