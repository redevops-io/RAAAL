"""`/auth/login`, `/auth/callback`, `/auth/logout`.

Mounted always and *served* only where the deployment declares an issuer. A
build with no provider answers these with a refusal that says so, rather than
404 — the two are different facts, and a 404 reads as "this application has no
login" when what is true is "this deployment has not been given one".

No page here trusts anything it was handed. The callback verifies the token
before a cookie is written, and the cookie is verified again on every request
that reads it.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..deploy.identity import Identity, IdentityUnavailable
from ..deploy.login import (FLOW_COOKIE, SESSION_COOKIE, Flow, LoginFailed,
                            begin, complete, flow_cookie, session_cookie,
                            viewer)

router = APIRouter(prefix="/auth", tags=["auth"])


def _target():
    from ..deploy.context import current

    return current().identity


def redirect_uri(request: Request) -> str:
    """Where the provider sends the browser back.

    Built from the configured application hostname rather than from the
    request's own Host header. A redirect URI taken from the request is one an
    attacker can influence, and it must match what was registered exactly — so
    deriving it from configuration is both the safe answer and the one that
    matches.
    """
    base = (_target().public_base_url or "").rstrip("/")
    if base:
        return f"{base}/auth/callback"
    return str(request.url_for("auth_callback"))


def signed_in(request: Request) -> Optional[Identity]:
    """The verified viewer of this request, or `None`."""
    target = _target()
    return viewer(request.cookies.get(SESSION_COOKIE),
                  issuer=target.issuer, audience=target.audience)


def _refusal(message: str, *, status: int = 400) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8>"
        "<title>Sign in — Quantify</title>"
        "<h1>Not signed in</h1>"
        f"<p>{message}</p>"
        '<p><a href="/auth/login">Try again</a> · <a href="/">Home</a></p>',
        status_code=status)


@router.get("/login", name="auth_login")
def login(request: Request, next: str = "/workspace"):
    target = _target()
    if not target.configured:
        return _refusal(
            "This deployment has no identity provider configured, so there is "
            "nothing to sign in to. That is a deployment setting, not "
            "something you can fix from here.", status=503)

    # Only a path on this site. An open redirect turns a login link into a way
    # to send somebody somewhere else with this site's name on it.
    destination = next if next.startswith("/") and not next.startswith("//") \
        else "/workspace"
    try:
        where, flow = begin(issuer=target.issuer, client_id=target.client_id,
                            redirect_uri=redirect_uri(request),
                            destination=destination)
    except IdentityUnavailable as refusal:
        return _refusal(str(refusal), status=503)
    except Exception as error:  # noqa: BLE001
        return _refusal(
            "The identity provider could not be reached, so signing in is not "
            f"possible right now ({type(error).__name__}).", status=502)

    response = RedirectResponse(where, status_code=303)
    response.set_cookie(**flow_cookie(flow))
    return response


@router.get("/callback", name="auth_callback")
def callback(request: Request, code: str = "", state: str = "",
             error: str = "", error_description: str = ""):
    target = _target()
    if not target.configured:
        return _refusal("This deployment has no identity provider.", status=503)

    if error:
        # The provider declined. Its own words, because "access_denied" after
        # somebody pressed Cancel should not read as a fault.
        return _refusal(
            f"The identity provider did not complete the sign-in: "
            f"{error_description or error}.")

    raw = request.cookies.get(FLOW_COOKIE)
    if not raw:
        return _refusal(
            "This sign-in could not be matched to one that started here — the "
            "browser sent no login cookie. Starting again is the fix.")

    try:
        flow = Flow.from_cookie(raw)
        who, token = complete(issuer=target.issuer,
                              client_id=target.client_id,
                              audience=target.audience,
                              redirect_uri=redirect_uri(request),
                              code=code, state=state, flow=flow)
    except LoginFailed as refusal:
        return _refusal(str(refusal))
    except Exception as error:  # noqa: BLE001
        return _refusal(
            "The sign-in could not be verified, so no session was created "
            f"({type(error).__name__}).", status=502)

    response = RedirectResponse(flow.destination, status_code=303)
    response.set_cookie(**session_cookie(token))
    response.delete_cookie(FLOW_COOKIE, path="/")
    return response


@router.get("/logout", name="auth_logout")
def logout(request: Request):
    """Ends the session here.

    Deliberately local. Redirecting to the provider's end-session endpoint
    would sign the person out of every application sharing it, which is not
    what pressing "sign out" on this one asks for.
    """
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
