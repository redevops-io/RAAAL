"""Who is asking, established by a token this deployment can verify.

The workspace has been behind one shared HTTP basic credential since the pilot
started. That is a door, not an identity: everybody who has it is the same
person as far as the application is concerned, and a plan cannot belong to
anybody in particular.

This module answers one question — *whose token is this* — and answers it only
when the answer is provable. Every path that cannot verify returns `None` and
says why. There is no branch that trusts an unverified claim, because the whole
value of a token over a shared password is that it was signed by something the
application checked.

**Verification is not hand-rolled.** Parsing a JWT is easy and verifying one
correctly is not: the failure modes are accepting `alg: none`, skipping the
audience, trusting the `kid` a token names without confirming the issuer
published it, and forgetting that an expired token is still perfectly
well-formed. PyJWT does the cryptography and this module supplies the policy.

**Four checks, and none of them optional.** Signature against a key the issuer
publishes, `iss` matching the declared issuer, `aud` matching the declared
audience, and expiry. A deployment that has not declared an issuer and audience
does not get a relaxed check — it gets no identity at all, which is what
`configured` reports and what the surface refuses on.
"""
from __future__ import annotations

import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

#: How long a fetched key set is reused. Issuers rotate signing keys, and a
#: cache with no expiry turns a rotation into an outage that looks like every
#: token suddenly being invalid.
JWKS_TTL_SECONDS = 3600

ALGORITHMS = ("RS256",)

#: How far apart the application's clock and the issuer's may drift before a
#: token that is genuinely valid reads as expired (or not-yet-valid). Sixty
#: seconds is the usual allowance: it is far smaller than any token lifetime,
#: so it cannot keep an actually-expired token alive in any meaningful way,
#: and it stops a one-second skew between two machines from silently ending a
#: session the issuer still considers current.
LEEWAY_SECONDS = 60


@dataclass(frozen=True)
class Identity:
    """A verified subject. Never constructed from an unverified token."""

    subject: str
    email: str = ""
    name: str = ""
    issuer: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"subject": self.subject, "email": self.email,
                "name": self.name, "issuer": self.issuer}


class IdentityUnavailable(Exception):
    """The deployment cannot establish identity at all.

    Distinct from a rejected token. One says this build is not configured to
    know who anybody is; the other says this particular person is not who the
    token claims. Collapsing them would let a misconfiguration read as a room
    full of invalid users.
    """


def _internal(url: str, internal: str) -> str:
    """The same request, addressed to this deployment rather than the internet.

    The provider is a container beside the application, behind the same proxy
    that publishes it. Reaching it by its public name sends the request out to
    the edge and back down a tunnel — where Cloudflare answered a management
    call with `403 error code: 1010`, its bot rule, against a Python user
    agent. The provider never saw it.

    Only the *address* changes. The issuer stays what it is: `iss` is part of
    every token and is verified against the public name, so rewriting that
    would break the very check this module exists to perform.
    """
    if not internal:
        return url
    here, there = urlsplit(url), urlsplit(internal)
    return urlunsplit((there.scheme, there.netloc, here.path, here.query, ""))


def _headers(url: str, internal: str) -> Dict[str, str]:
    """What the proxy needs to route an internal request like a public one."""
    if not internal:
        return {}
    # Host selects the identity site; the scheme is the browser's, because TLS
    # terminates at the edge and the provider builds its issuer from what it is
    # told. Without it the provider advertises `http://` and no token verifies.
    return {"Host": urlsplit(url).netloc, "X-Forwarded-Proto": "https"}


def _read(url: str, *, internal: str = "", timeout: float = 10.0) -> Any:
    import json

    request = urllib.request.Request(_internal(url, internal))
    for name, value in _headers(url, internal).items():
        request.add_header(name, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class _Keys:
    """The issuer's published signing keys, fetched once and reused."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_issuer: Dict[str, Any] = {}
        self._fetched_at: Dict[str, float] = {}

    def clear(self) -> None:
        with self._lock:
            self._by_issuer.clear()
            self._fetched_at.clear()

    def client(self, issuer: str, *, internal: str = "",
               timeout: float = 10.0):
        from jwt import PyJWKClient

        key = f"{issuer}|{internal}"
        with self._lock:
            fresh = (time.monotonic() - self._fetched_at.get(key, 0.0)
                     < JWKS_TTL_SECONDS)
            if key in self._by_issuer and fresh:
                return self._by_issuer[key]

        uri = discovery(issuer, internal=internal, timeout=timeout)["jwks_uri"]
        client = PyJWKClient(_internal(uri, internal), cache_keys=True,
                             headers=_headers(uri, internal) or None)
        with self._lock:
            self._by_issuer[key] = client
            self._fetched_at[key] = time.monotonic()
        return client


KEYS = _Keys()


def discovery(issuer: str, *, internal: str = "",
              timeout: float = 10.0) -> Mapping[str, Any]:
    """The issuer's OIDC discovery document.

    Read from the issuer rather than assembled from a template. Providers do
    not agree on where their key set lives, and a URL guessed from the issuer
    is a guess that works until the day the provider moves it.
    """
    return _read(f"{issuer.rstrip('/')}/.well-known/openid-configuration",
                 internal=internal, timeout=timeout)


def verify(token: str, *, issuer: str, audience: str, internal: str = "",
           timeout: float = 10.0) -> Identity:
    """The subject this token proves, or an exception naming what failed.

    Raises rather than returning `None` so a caller cannot treat a failure as
    an anonymous visitor by forgetting to check. The distinction matters: an
    invalid token is somebody attempting to be someone, and a missing one is
    somebody who has not tried yet.
    """
    import jwt

    if not issuer or not audience:
        raise IdentityUnavailable(
            "this deployment declares no OIDC issuer and audience, so it "
            "cannot establish who anybody is")

    signing = KEYS.client(issuer, internal=internal,
                          timeout=timeout).get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token, signing.key, algorithms=list(ALGORITHMS),
        audience=audience, issuer=issuer,
        # A small tolerance on the time-based checks (exp, and iat/nbf if
        # present). Two machines never agree on the second, and without this a
        # clock a minute ahead of the issuer would reject a token the issuer
        # only just minted — a valid session reading as an invalid one purely
        # because of skew. The allowance is orders of magnitude below any token
        # lifetime, so it does not weaken expiry; it only removes the false
        # negative at the boundary.
        leeway=LEEWAY_SECONDS,
        # Spelled out rather than left to defaults. A future default that
        # relaxes one of these would relax it here silently, and the whole
        # point of this module is that none of the four is optional.
        options={"require": ["exp", "iss", "aud", "sub"],
                 "verify_signature": True, "verify_exp": True,
                 "verify_aud": True, "verify_iss": True},
    )
    return Identity(subject=str(claims["sub"]),
                    email=str(claims.get("email") or ""),
                    name=str(claims.get("name")
                             or claims.get("preferred_username") or ""),
                    issuer=str(claims["iss"]))


def configured() -> bool:
    """Whether this deployment can establish identity at all.

    Asked by the surface before it offers a login. A build with no issuer is
    not a build where everybody is anonymous — it is one that must not pretend
    to have accounts.
    """
    from .context import current

    identity = current().identity
    return bool(identity.issuer and identity.audience)
