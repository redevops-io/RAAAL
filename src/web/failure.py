"""The single boundary where a failure becomes a response.

    DatabaseFailure -> status + {code, message, retryable, request_id}
                    -> operator log keeps sqlstate, reason, driver text

Installed on the application rather than written into handlers. A route that
catches its own database exception is a route that decides, alone, what a
caller learns — and the decision that gets it wrong is the one on the path
nobody tested. There are 49 handlers; there is one of these.

**The response is built from the failure's `public()` and nothing else.** No
handler composes a message from an exception, because a message composed from
an exception carries whatever that exception carried. `tests/test_error_surface.py`
enumerates the boundaries and fails on one that reaches for a driver exception
itself.

**Correlation, not disclosure.** A caller gets a `request_id` and the operator
log gets the same id beside the SQLSTATE, the constraint and the driver text.
That is what makes a support conversation possible without the response
carrying the schema: the user quotes an opaque id, and the operator finds the
detail that was never sent.
"""
from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

LOG = logging.getLogger("uvicorn.error")

#: The header a caller may supply to correlate their own retries, and the one
#: echoed back. Generated when absent — an id that only exists when the client
#: thought to send one is an id missing from exactly the incidents worth
#: investigating.
REQUEST_ID_HEADER = "X-Request-ID"


def _record(message: str, payload: dict) -> None:
    """Write to the operator channel without letting it take a response down.

    A handler can be unavailable, a formatter can raise, a field can be
    unserializable — none of which is the caller's problem. Without this, a
    broken log sink turned a correct 409 into an unhandled 500: the private
    channel failing cost the caller their answer *and* replaced a precise
    result with a generic one.

    The fallback is `stderr` directly. It is the last place that can still say
    something when the logging subsystem itself is the thing that failed, and
    saying nothing would leave the incident with no record at all — which is
    the same silence the migration defect produced, arrived at differently.
    """
    try:
        LOG.error("%s %s", message, payload)
        return
    except Exception as exc:                                     # noqa: BLE001
        pass
    try:
        print(f"[operator-log-unavailable] {message} {payload}",
              file=sys.stderr, flush=True)
    except Exception:                                            # noqa: BLE001
        # Nothing further is available. The response still goes out, which is
        # the decision this whole function encodes.
        pass


def request_id_for(request: Any) -> str:
    supplied = ""
    try:
        supplied = request.headers.get(REQUEST_ID_HEADER, "") or ""
    except Exception:                                            # noqa: BLE001
        supplied = ""
    # Bounded and stripped of anything that is not an identifier: a client
    # controls this value, and it is written to the operator log.
    cleaned = "".join(one for one in supplied if one.isalnum() or one in "-_")[:64]
    return cleaned or f"req-{uuid.uuid4().hex[:16]}"


def install(app) -> None:
    """Route every database failure through one translation."""
    from fastapi.responses import JSONResponse

    from ..db.errors import DatabaseFailure
    from ..workspace.apply import ProposalConflict, TransitionIntegrityError

    #: Registered individually rather than by a shared base: `ProposalConflict`
    #: is an `ApplyRefused` and most refusals are ordinary 422s that should keep
    #: their own message. Only the two that carry a public category belong here.
    _refusals = (ProposalConflict, TransitionIntegrityError)

    @app.exception_handler(DatabaseFailure)
    async def _database_failure(request, failure: DatabaseFailure):
        request_id = failure.request_id or request_id_for(request)

        # Private first, so a failure in serializing the response cannot lose
        # the only record of what happened.
        _record("database failure", {**failure.private(),
                                     "request_id": request_id,
                                     "path": _safe_path(request)})

        payload = failure.public()
        payload["request_id"] = request_id
        return JSONResponse(payload, status_code=failure.status,
                            headers={REQUEST_ID_HEADER: request_id})

    async def _transition_refusal(request, refusal):
        """Application refusals that carry a public category.

        A conditional transition that moved no row is not a driver failure —
        the statement succeeded — and it is not an ordinary refusal either: the
        caller must re-read before retrying. `apply.py` names the category
        beside the exception, because the layer that knows how many rows moved
        is the layer that knows what the caller must do, and a route inferring
        that from a class name would be deciding it a second time.
        """
        from ..db.errors import DatabaseFailure
        from ..db.errors import Classification, InternalReason

        request_id = request_id_for(request)
        failure = DatabaseFailure(
            Classification(refusal.public_code, InternalReason.NO_ROW_TRANSITIONED
                           if refusal.retry_disposition.retryable
                           else InternalReason.TOO_MANY_ROWS_TRANSITIONED,
                           refusal.retry_disposition),
            request_id=request_id)
        # The refusal's own message is application-authored and safe, but it is
        # not sent: one envelope, one vocabulary, and a message that varies by
        # instance is a message somebody will eventually build from state.
        _record("transition refusal",
                {**failure.private(), "detail": str(refusal),
                 "path": _safe_path(request)})
        payload = failure.public()
        payload["request_id"] = request_id
        return JSONResponse(payload, status_code=failure.status,
                            headers={REQUEST_ID_HEADER: request_id})

    for refusal_type in _refusals:
        app.add_exception_handler(refusal_type, _transition_refusal)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        """Anything that reaches here is a bug, and its text is still a
        deployment detail.

        FastAPI's default returns a bare 500 and logs a traceback, which is
        already safe. This exists so the response carries the same correlation
        id as the log line — without it, the one class of failure nobody
        anticipated is also the one with no way to look it up.
        """
        request_id = request_id_for(request)
        _record("unhandled failure",
                {"request_id": request_id, "path": _safe_path(request),
                 "exception": type(exc).__name__})
        return JSONResponse(
            {"code": "INTERNAL_ERROR",
             "message": "The request could not be completed.",
             "retryable": False, "request_id": request_id},
            status_code=500, headers={REQUEST_ID_HEADER: request_id})


def _safe_path(request: Any) -> str:
    """The route template, not the resolved path.

    `/workspace/plans/{plan_id}` rather than `/workspace/plans/my-roth-ira`:
    the path carries user-chosen identifiers, and the operator log is a second
    place they would then have to be redacted from.
    """
    try:
        route = request.scope.get("route")
        return getattr(route, "path", "") or ""
    except Exception:                                            # noqa: BLE001
        return ""
