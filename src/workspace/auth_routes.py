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

# Imported under an alias, not as `login`: the `login` route handler below
# would otherwise shadow the module, and `signed_in` would be calling the
# view function instead of `login_flow.refresh`.
from ..deploy import login as login_flow
from ..deploy.identity import Identity, IdentityUnavailable
from ..deploy.login import (FLOW_COOKIE, REFRESH_COOKIE, SESSION_COOKIE, Flow,
                            LoginFailed, begin, complete, flow_cookie,
                            refresh_cookie, session_cookie, viewer)

router = APIRouter(prefix="/auth", tags=["auth"])

#: Distinguishes "this request's viewer has not been resolved yet" from "it was
#: resolved and the answer was nobody (`None`)". Without it, caching a `None`
#: decision and re-reading it would be indistinguishable from never having run,
#: and `signed_in` would refresh again on the second call within one request.
_UNSET = object()


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
    """The verified viewer of this request, or `None`, refreshing if it can.

    Resolves in three steps, and each request resolves *once*: the outcome is
    cached on `request.state`, because both the private-surface gate and the
    `_note_the_viewer` middleware ask this question and a refresh must not be
    attempted twice (and its rotated token overwritten) within one request.

    1. If the session cookie still verifies, that is the viewer.
    2. If it does not but a refresh cookie is present and this deployment is
       configured, exchange the refresh token for a new ID token. On success the
       new tokens are stashed on `request.state` for `_apply_refreshed_session`
       in `api.py` to write onto the response, and the renewed identity is the
       viewer. On failure the stale cookies are marked for deletion.
    3. If the session cookie is present but there is nothing to refresh it with,
       the stale cookies are likewise marked for deletion.

    Marking the stale cookies for deletion is what makes the failure *fail
    closed* rather than silently. `owner.current()` treats a request with no
    verifiable session as the shared `"pilot"` workspace, so a logged-in person
    whose token expired would otherwise have their next save land under `pilot`
    — someone else's data. Clearing the cookies means the very next request is
    genuinely session-less (bounced to sign-in on a gated page, legitimately
    anonymous elsewhere) instead of carrying a dead token that reads as `pilot`.

    Never raises: the gate and `_note_the_viewer` both depend on this returning
    `Optional[Identity]` and never interrupting a page. A refresh that errors is
    treated as a refresh that failed.
    """
    decided = getattr(request.state, "_viewer_cached", _UNSET)
    if decided is not _UNSET:
        return decided

    target = _target()
    cookie = request.cookies.get(SESSION_COOKIE)
    identity = viewer(cookie, issuer=target.issuer, audience=target.audience,
                      internal=target.internal_base_url)
    if identity is not None:
        request.state._viewer_cached = identity
        return identity

    # The session did not resolve. It is expired, malformed, or absent. If a
    # refresh token is present and this deployment can talk to a provider, try
    # to renew rather than treating the person as signed out.
    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if refresh_token and target.configured:
        try:
            renewed, new_session, new_refresh = login_flow.refresh(
                issuer=target.issuer, client_id=target.client_id,
                audience=target.audience, refresh_token=refresh_token,
                internal=target.internal_base_url)
        except Exception:  # noqa: BLE001 - any refresh failure is "not renewable"
            request.state._clear_session = True
            request.state._viewer_cached = None
            return None
        request.state._new_session_token = new_session
        request.state._new_refresh_token = new_refresh
        request.state._viewer_cached = renewed
        return renewed

    # A session cookie that no longer verifies and cannot be refreshed must not
    # be left in place to read as `pilot` on the next request — clear it. A
    # request that never carried a session is left alone: there is nothing stale
    # to remove, and it is legitimately anonymous.
    if cookie:
        request.state._clear_session = True
    request.state._viewer_cached = None
    return None


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
                            destination=destination,
                            internal=target.internal_base_url)
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
        who, token, refresh_token = complete(issuer=target.issuer,
                                             client_id=target.client_id,
                                             audience=target.audience,
                                             redirect_uri=redirect_uri(request),
                                             code=code, state=state, flow=flow,
                                             internal=target.internal_base_url)
    except LoginFailed as refusal:
        return _refusal(str(refusal))
    except Exception as error:  # noqa: BLE001
        return _refusal(
            "The sign-in could not be verified, so no session was created "
            f"({type(error).__name__}).", status=502)

    response = RedirectResponse(flow.destination, status_code=303)
    response.set_cookie(**session_cookie(token))
    # The refresh cookie is only written when the provider actually returned a
    # refresh token. A provider configured without `offline_access` yields an
    # empty one, and persisting a blank credential would only produce a failed
    # refresh later — the session still works, it simply cannot renew itself.
    if refresh_token:
        response.set_cookie(**refresh_cookie(refresh_token))
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
    # Both halves of the session go. Leaving the refresh token behind would let
    # the very next request mint a new session and undo the sign-out.
    response.delete_cookie(REFRESH_COOKIE, path="/")
    return response
