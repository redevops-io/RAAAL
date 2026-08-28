"""Abuse controls for the public evaluation surface (§11 of the strategy-lab plan).

Public evaluation is the product's front door, and a front door with no lock is an
abuse surface: an anonymous visitor can post a prompt as fast as a script can
send one, as large as a socket will carry, and — if the evaluator were careless —
containing a URL it hoped the server would fetch. §11 lists the controls that
close that surface without closing the door on a legitimate visitor.

**Everything here is additive, generous by default, and fails open.** This is a
live public service (quantify.club). A control that blocked a real visitor, or a
limiter that took the site down when its own state got into a bad shape, would be
a worse outcome than the abuse it was guarding against. So every threshold has a
sane default, every threshold is an env knob, and every guard that could raise is
wrapped so its failure serves the request rather than refusing it.

The controls, and the knobs that tune them:

    Content-Security-Policy        QUANTIFY_CSP (full override)
    IP/session rate limit          QUANTIFY_RATE_LIMIT_PER_MIN (default 60)
                                   QUANTIFY_RATE_LIMIT_WINDOW_SECONDS (default 60)
    request-size ceiling           QUANTIFY_MAX_BODY_BYTES (default 65536 = 64 KB)
    prompt length cap              QUANTIFY_MAX_PROMPT_CHARS (default 4000)
    per-evaluation model budget    QUANTIFY_MODEL_BUDGET_CEILING (default 1)
    CSRF on the authenticated save QUANTIFY_CSRF_ENFORCE (default off)

None of this touches research or static: the rate limit and the size ceiling are
scoped to the evaluation POST endpoints by `is_rate_limited`, so the daily
dashboard and the published library are never throttled.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from hmac import compare_digest
from secrets import token_urlsafe
from typing import Dict, Optional, Tuple

LOG = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """An integer knob, read at call time so a redeploy's env takes effect.

    Fails open to the default on anything unparseable: a malformed limit must
    not become a limit of zero, which would refuse every request.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        LOG.warning("ignoring unparseable %s=%r; using %d", name, raw, default)
        return default
    return value if value > 0 else default


# --- 1. Content-Security-Policy -------------------------------------------
#
# The one control that can blank the site if it is wrong, so it is written
# against what the current pages actually need rather than against a textbook
# ideal. The pages use inline event handlers (`onclick="toggleTheme()"`), inline
# `<style>` blocks and `style=` attributes, and the research dashboard is a
# standalone Bokeh document that both carries inline scripts and loads BokehJS
# from cdn.bokeh.org. A strict `script-src 'self'` would break every one of
# those, so the policy is deliberately permissive where the templates require it
# and tight everywhere else.
#
# `'unsafe-inline'` is the pragmatic choice, not the final one: the correct
# tightening is a per-response nonce threaded through every inline handler and
# `<script>`/`<style>` in the templates, which is a template migration in its own
# right. Until that is done, allowing inline keeps the UI working; the directive
# is a single env override (`QUANTIFY_CSP`) away from being replaced wholesale by
# a deployment that has done the nonce work.

#: Where the research dashboard's BokehJS bundles come from. Named once so the
#: script/connect directives cannot disagree about it.
_BOKEH_CDN = "https://cdn.bokeh.org"

DEFAULT_CSP = "; ".join((
    # Nothing loads from anywhere unless a more specific directive allows it.
    "default-src 'self'",
    # Inline handlers + inline <script> blocks (the theme toggle, the pilot
    # page's chart bootstrap) and the Bokeh bundles the /research document pulls.
    f"script-src 'self' 'unsafe-inline' {_BOKEH_CDN}",
    # Inline <style> blocks in every base.html and the pervasive style=
    # attributes across the workspace templates.
    "style-src 'self' 'unsafe-inline'",
    # Bokeh renders to canvas and can inline raster exports as data: URIs; a
    # page favicon is also a data: URI. No remote image host is used.
    "img-src 'self' data:",
    # System-font stack only; data: kept for any inlined face without opening a
    # remote font host.
    "font-src 'self' data:",
    # No cross-origin XHR/fetch/WebSocket is used — the standalone Bokeh document
    # does not phone home — so same-origin only.
    "connect-src 'self'",
    # Belt-and-braces with the existing X-Frame-Options: DENY header.
    "frame-ancestors 'none'",
    # Forms post to same-origin routes (/evaluate, /pilot/answer, the saves);
    # the OIDC hop is a top-level redirect, not a form submission, so this does
    # not touch login.
    "form-action 'self'",
    # <base> injection and plugin embeds have no legitimate use here.
    "base-uri 'self'",
    "object-src 'none'",
))


def content_security_policy() -> str:
    """The CSP header value for every response.

    A deployment that has done the nonce work can replace the whole policy with
    `QUANTIFY_CSP`; otherwise the default above, which is written to keep the
    current pages rendering.
    """
    override = os.environ.get("QUANTIFY_CSP")
    return override if override else DEFAULT_CSP


# --- 2. IP/session rate limit ---------------------------------------------
#
# A fixed-window counter keyed by client identity. In-process on purpose: it is
# a courtesy speed bump for a single instance, not a distributed quota, and a
# per-instance bump with a generous default is worth far more than the operational
# weight of a shared store for what it defends. Behind Cloudflare the real client
# is the leftmost X-Forwarded-For hop; the socket peer is the fallback.


def client_ip(request) -> str:
    """The caller's identity for rate limiting.

    Prefers the leftmost `X-Forwarded-For` hop because the deployment sits behind
    Cloudflare (every hostname is HTTPS-only there), where the socket peer is the
    proxy and not the visitor. Falls back to the socket peer for a direct
    connection or a test client.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or "unknown"


class FixedWindowLimiter:
    """A per-key fixed-window counter. Thread-safe, in-process, best-effort."""

    def __init__(self) -> None:
        self._hits: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, limit: int, window_seconds: int,
              now: Optional[float] = None) -> bool:
        """Whether this hit is within `limit` per `window_seconds` for `key`.

        Raises nothing a caller must catch under normal use; the middleware still
        wraps it, because a broken limiter must never be the reason a page fails.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= window_seconds:
                start, count = now, 0
            count += 1
            self._hits[key] = (start, count)
            return count <= limit

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


#: One limiter for the process. Reset between tests via `reset_rate_limits`.
RATE_LIMITER = FixedWindowLimiter()


def reset_rate_limits() -> None:
    RATE_LIMITER.reset()


def rate_limit_per_minute() -> int:
    return _int_env("QUANTIFY_RATE_LIMIT_PER_MIN", 60)


def rate_limit_window_seconds() -> int:
    return _int_env("QUANTIFY_RATE_LIMIT_WINDOW_SECONDS", 60)


#: The evaluation endpoints the limit and the size ceiling apply to. Research,
#: static, health and the library are deliberately absent — throttling published
#: research would be a different product.
RATE_LIMITED_PREFIXES = ("/evaluate", "/pilot/answer", "/pilot/save")


def is_rate_limited(path: str) -> bool:
    """Whether abuse controls apply to this path. Evaluation only."""
    return any(path == p or path.startswith(p + "/") or path.startswith(p)
               for p in RATE_LIMITED_PREFIXES) and not path.startswith("/pilot/save/resume")


def within_rate_limit(request) -> bool:
    """Best-effort limit check. Fails **open** on any internal error.

    The `try` is the whole point of the control being safe on a live service: a
    limiter that raised — a key type it did not expect, a clock going backwards —
    would otherwise turn every request into a 500. Here a broken limiter simply
    stops limiting, which is the failure that keeps the site up.
    """
    try:
        return RATE_LIMITER.allow(
            client_ip(request),
            limit=rate_limit_per_minute(),
            window_seconds=rate_limit_window_seconds())
    except Exception:  # noqa: BLE001 - a broken limiter must not fail the request
        LOG.exception("rate limiter failed; serving the request (fail-open)")
        return True


# --- 3. request-size ceiling ----------------------------------------------


def max_body_bytes() -> int:
    return _int_env("QUANTIFY_MAX_BODY_BYTES", 64 * 1024)


def oversized(request) -> bool:
    """Whether the declared body is over the ceiling.

    Reads `Content-Length` — the size the client announced — which is enough to
    reject an oversized POST before the body is drawn into memory. A request that
    lies about its length still meets the same cap when the framework parses the
    form; this is the cheap first gate.
    """
    declared = request.headers.get("content-length")
    if not declared:
        return False
    try:
        return int(declared) > max_body_bytes()
    except (TypeError, ValueError):
        return False


# --- 4. prompt normalization + model-budget ceiling -----------------------


def max_prompt_chars() -> int:
    return _int_env("QUANTIFY_MAX_PROMPT_CHARS", 4000)


_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """Trim, collapse runs of whitespace, and cap the length.

    A normalization, not a rewrite: an ordinary sentence ("invest $500 monthly")
    is returned unchanged, so a normal evaluation lands on exactly the same
    content-addressed identity it did before. What it removes is the pathological
    input — a megabyte of spaces, a prompt padded past the length cap — that only
    an abuser sends.
    """
    if not text:
        return text
    collapsed = _WHITESPACE.sub(" ", text).strip()
    cap = max_prompt_chars()
    return collapsed[:cap] if len(collapsed) > cap else collapsed


def model_budget_ceiling() -> int:
    """The per-evaluation cap on model calls.

    An anonymous evaluation should cost a bounded amount of model work — one
    stage-1 call at most on the current path. This is the ceiling the evaluation
    path checks through `within_model_budget`; a deployment that wires a real
    per-request counter enforces it against a live count, and until then the
    length cap above is what bounds the work a single prompt can demand.
    """
    return _int_env("QUANTIFY_MODEL_BUDGET_CEILING", 1)


def within_model_budget(model_calls: int) -> bool:
    """Whether an evaluation that made `model_calls` calls is within budget.

    The hook §11 asks for. It is deliberately a pure predicate rather than a piece
    of global state: the evaluation path can call it with whatever count it holds,
    and a deployment with a real accounting layer can pass that layer's number.
    """
    return model_calls <= model_budget_ceiling()


_URL = re.compile(r"https?://|ftp://|file://", re.IGNORECASE)


def contains_url(text: str) -> bool:
    """Whether a prompt mentions a URL scheme.

    Used only to *log* that a prompt looked like it wanted a fetch, never to act
    on it. The evaluation path treats a prompt as text end to end — it hands the
    sentence to the parser and never dereferences anything inside it — so a URL in
    a prompt is inert. This helper exists so a test can assert that inertness and
    so the log can note the attempt; there is no code path that fetches it.
    """
    return bool(text and _URL.search(text))


# --- 5. CSRF on the authenticated save ------------------------------------
#
# Double-submit cookie: the server issues an unguessable token in a cookie and
# renders the same token as a hidden field in the save forms; a state-changing
# POST must carry both, matching. A cross-site page can drive the browser to POST
# but cannot read the cookie to echo it in the body, so its forged POST fails the
# match.
#
# **Off by default, and safe with it.** The live save is already CSRF-resistant
# without this: the session cookie is SameSite=Lax (a cross-site POST carries no
# session), and the anonymous save completes through a GET that consumes an
# unguessable single-use token bound to the session. This adds defense in depth a
# deployment turns on with `QUANTIFY_CSRF_ENFORCE=1`; leaving it off keeps every
# existing caller working unchanged.

CSRF_COOKIE = "quantify_csrf"
CSRF_FIELD = "csrf_token"

#: The token for the request being rendered, filled by the CSRF middleware in
#: `src.api` and read by the `csrf_token()` template global. A ContextVar for the
#: same reason `routes._LOOKING` is one: Jinja globals take no request, and this
#: is the one place the request's token is in scope.
from contextvars import ContextVar  # noqa: E402

CSRF_TOKEN: "ContextVar[str]" = ContextVar("quantify_csrf_token", default="")


def current_csrf_token() -> str:
    """The CSRF token to render into a save form's hidden field, or ``""``."""
    return CSRF_TOKEN.get()


def csrf_enforced() -> bool:
    return os.environ.get("QUANTIFY_CSRF_ENFORCE", "").strip().lower() in (
        "1", "true", "yes", "on")


def issue_csrf_token() -> str:
    return "csrf-" + token_urlsafe(24)


def csrf_cookie(token: str) -> Dict[str, object]:
    """Cookie attributes for the CSRF token.

    HttpOnly: the token is rendered into the form by the server, so no page
    script needs to read it, and keeping it out of JS removes one way it leaks.
    Secure + SameSite=Lax match the session cookie so the two travel together.
    """
    return {"key": CSRF_COOKIE, "value": token, "httponly": True,
            "secure": True, "samesite": "lax", "path": "/"}


def verify_csrf(request, submitted: str) -> bool:
    """Whether the submitted token matches the one in the cookie.

    Constant-time, and false whenever either side is missing: an absent cookie or
    an absent field is a failed check, not a skipped one, so enforcement cannot be
    bypassed by simply sending neither.
    """
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not cookie or not submitted:
        return False
    try:
        return compare_digest(str(cookie), str(submitted))
    except Exception:  # noqa: BLE001
        return False


# --- 7. structured logging without sensitive bodies -----------------------


def log_event(event: str, request, *, outcome: str = "", **fields) -> None:
    """One structured line for an abuse-relevant event. Never the prompt body.

    Records what an operator needs to see an attack — the path, the method, the
    client, the outcome and whatever ids/counts the caller passes — and nothing
    that would put a person's strategy prose into the operational log (§11). It
    never raises into a request: logging is the expendable half.
    """
    try:
        safe = {k: v for k, v in fields.items()
                if k not in ("prompt", "describe", "text", "body", "sentence")}
        LOG.info(
            "abuse-event %s path=%s method=%s client=%s outcome=%s %s",
            event, getattr(request.url, "path", ""), request.method,
            client_ip(request), outcome,
            " ".join(f"{k}={v}" for k, v in sorted(safe.items())))
    except Exception:  # noqa: BLE001 - a log line must not be able to fail a page
        pass
