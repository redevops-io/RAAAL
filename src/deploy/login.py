"""The login flow: authorization code with PKCE, and a session that is a token.

`identity.py` answers *whose token is this*. This answers *how does somebody get
one*, and *what carries it between requests*.

**The session is the ID token, verified on every request.** There is no session
table and no server-side store. That is not a shortcut — it removes the two
things that go wrong with sessions: a secret to sign them with, which becomes a
third thing to generate, protect and rotate, and a store whose entries outlive
the token they were minted from. A cookie holding a JWT expires exactly when
the token does, because it is the token.

**PKCE, and no client secret.** The authorization code is bound to the request
that started it by a verifier only this browser holds, so a code intercepted on
the way back is useless. That is the property a client secret would otherwise
be providing, and this one does not have to be stored anywhere.

**The verifier rides in a cookie, not in memory.** A dictionary keyed by state
would be per-process — correct on one worker and a coin toss on two — and would
lose every login in flight on restart.

Every failure here is explicit. A callback that cannot be verified renders a
refusal naming which check failed; it never falls through to a session.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .identity import (Identity, IdentityUnavailable, _headers, _internal,
                       discovery, verify)

#: The cookie carrying the verified token. Read on every request.
SESSION_COOKIE = "quantify_session"

#: The cookie carrying the in-flight PKCE verifier and CSRF state. Short-lived:
#: it is only meaningful between the redirect out and the redirect back.
FLOW_COOKIE = "quantify_login"

FLOW_SECONDS = 600

#: What the application asks for. `openid` is required; the other two are what
#: the workspace shows a person about themselves. Nothing else is requested —
#: a scope that is not used is a permission asked for no reason.
SCOPES = "openid profile email"


class LoginFailed(Exception):
    """The callback could not be turned into a verified session.

    Carries a sentence for the person and never the underlying token. A failure
    here is either a misconfiguration or an attempt, and neither is improved by
    showing the artefact.
    """


@dataclass(frozen=True)
class Flow:
    """One login in flight."""

    state: str
    verifier: str
    destination: str = "/workspace"

    def to_cookie(self) -> str:
        return base64.urlsafe_b64encode(json.dumps(
            {"state": self.state, "verifier": self.verifier,
             "destination": self.destination}).encode()).decode()

    @classmethod
    def from_cookie(cls, raw: str) -> "Flow":
        try:
            found = json.loads(base64.urlsafe_b64decode(raw.encode()))
            return cls(state=str(found["state"]),
                       verifier=str(found["verifier"]),
                       destination=str(found.get("destination") or "/workspace"))
        except Exception as error:  # noqa: BLE001 - any malformed cookie is one case
            raise LoginFailed(
                "This login could not be matched to one that started here. "
                "That happens when the browser dropped the cookie, when the "
                "sign-in took longer than ten minutes, or when the link was "
                "not started on this site. Starting again is the fix."
            ) from error


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def begin(*, issuer: str, client_id: str, redirect_uri: str,
          destination: str = "/workspace", internal: str = "",
          timeout: float = 10.0) -> tuple[str, Flow]:
    """Where to send the browser, and what to remember while it is away."""
    if not issuer or not client_id:
        raise IdentityUnavailable(
            "this deployment declares no identity provider, so there is "
            "nothing to log in to")

    flow = Flow(state=secrets.token_urlsafe(24),
                # 32 bytes, well inside the 43–128 character range the spec
                # requires once base64url-encoded.
                verifier=secrets.token_urlsafe(32),
                destination=destination)
    query = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": flow.state,
        "code_challenge": _challenge(flow.verifier),
        "code_challenge_method": "S256",
    })
    # Read through the internal route, used as published. This URL is where a
    # *browser* is sent, so it must be the public one — internalising it would
    # send somebody's browser to a hostname only this network can resolve.
    endpoint = discovery(issuer, internal=internal,
                         timeout=timeout)["authorization_endpoint"]
    return f"{endpoint}?{query}", flow


def _exchange(*, issuer: str, client_id: str, redirect_uri: str, code: str,
              verifier: str, internal: str = "",
              timeout: float = 10.0) -> Mapping[str, Any]:
    endpoint = discovery(issuer, internal=internal,
                         timeout=timeout)["token_endpoint"]
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }).encode()
    # Unlike the authorization endpoint, nothing about this one involves a
    # browser: the application exchanges the code itself, so this call should
    # never leave the host.
    request = urllib.request.Request(
        _internal(endpoint, internal), data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 **_headers(endpoint, internal)},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # The provider's own words, which name the actual disagreement —
        # a redirect URI that does not match its registration, an expired
        # code, a verifier that does not hash to the challenge.
        detail = ""
        try:
            detail = json.loads(error.read()).get("error_description", "")
        except Exception:  # noqa: BLE001
            pass
        raise LoginFailed(
            "The identity provider would not exchange this sign-in for a "
            f"token{': ' + detail if detail else '.'}") from error


def complete(*, issuer: str, client_id: str, audience: str, redirect_uri: str,
             code: str, state: str, flow: Flow, internal: str = "",
             timeout: float = 10.0) -> tuple[Identity, str]:
    """The verified identity and the token to keep, or `LoginFailed`.

    The state comparison is constant-time and happens before anything is sent
    to the provider: a callback that did not start here should cost nothing.
    """
    if not secrets.compare_digest(state or "", flow.state):
        raise LoginFailed(
            "This sign-in did not start on this site, so it was not "
            "completed. Nothing was signed in.")

    granted = _exchange(issuer=issuer, client_id=client_id,
                        redirect_uri=redirect_uri, code=code,
                        verifier=flow.verifier, internal=internal,
                        timeout=timeout)
    token = granted.get("id_token")
    if not token:
        raise LoginFailed(
            "The identity provider returned no ID token, so there is nothing "
            "to establish who signed in. The application requests the "
            "`openid` scope, which is what obliges it to send one.")

    # The same verification every later request performs. Doing it here means a
    # token that would be rejected on the next page is rejected now, while
    # there is somebody to tell.
    return verify(token, issuer=issuer, audience=audience, internal=internal,
                  timeout=timeout), token


def viewer(token: Optional[str], *, issuer: str, audience: str,
           internal: str = "", timeout: float = 10.0) -> Optional[Identity]:
    """Who this request is from, or `None`.

    Returns `None` for every failure rather than raising, because the caller is
    a page deciding whether to show a login link. The distinction that matters
    — configured versus not — is asked separately, and an expired cookie is not
    an error worth interrupting a page for.
    """
    if not token or not issuer or not audience:
        return None
    try:
        return verify(token, issuer=issuer, audience=audience,
                      internal=internal, timeout=timeout)
    except Exception:  # noqa: BLE001 - an unverifiable cookie is simply not a viewer
        return None


def session_cookie(token: str) -> Dict[str, Any]:
    """Cookie attributes for the session. Kept here so both the login and the
    logout path spell them identically — a deletion that disagrees with the
    write on `path` leaves the cookie in place."""
    return {"key": SESSION_COOKIE, "value": token, "httponly": True,
            "secure": True, "samesite": "lax", "path": "/"}


def flow_cookie(flow: Flow) -> Dict[str, Any]:
    return {"key": FLOW_COOKIE, "value": flow.to_cookie(), "httponly": True,
            "secure": True, "samesite": "lax", "path": "/",
            "max_age": FLOW_SECONDS}
