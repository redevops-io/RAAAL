"""Who is doing this, enough to link one attempt to the next — and no more.

Five of the six things worth observing about a pilot participant are invisible
without an answer to "was that the same person as a moment ago":

    the exact prompt they entered        one row, or a chain of six revisions
    an unnecessary clarification         only visible if you see what they did next
    whether they edited and resubmitted  unanswerable with no linkage at all
    whether they reopened saved plans    already countable
    whether they went back to legacy     needs to know they were in the pilot

So there is a participant token. It is a random opaque string in a cookie. It
is **not** identity: it carries no name, no email, no address, and it says
nothing about the person except that a series of requests came from one browser.
That is exactly the amount needed to see a revision chain and no more, and it is
the reason this is a token rather than a login.

**Two stores, and the split is the point.**

    pilot_events        counts and codes, no prose, no permission needed
    pilot_transcripts   what someone actually typed, off unless declared

Prose in the events table would have broken the property the instrumentation
tests assert — that nothing countable moves when a sentence is reworded. But
the sentences themselves are the most valuable thing the pilot produces, because
"the runtime asked an unnecessary clarification" is a claim nobody can settle
from a counter. Keeping them apart lets both be true: the numbers stay
countable, the words stay verbatim, and the words are held under a declaration
rather than as a side effect of counting.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

#: Opaque, and named so it reads as what it is in a browser's cookie list.
COOKIE = "quantify_pilot_participant"

#: `owner` is the tenant and decides who may read the row. `participant` is the
#: study pseudonym, is not part of the key, and is nullable — so the pseudonym
#: can be cleared from a transcript later without destroying or re-keying what
#: was typed. There is no foreign key to an authenticated user; a study that
#: needs that mapping keeps it in its own narrowly-held table, where it can be
#: deleted on its own.
#:
#: A transcript is a record of words that were typed. Making the pseudonym part
#: of its storage identity would put deduplication, exports and restores on the
#: wrong side of a link that has to stay severable.
SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_transcripts (
    owner         TEXT NOT NULL,
    entry_id      TEXT NOT NULL,
    participant   TEXT,
    at            TEXT NOT NULL,
    attempt       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    detail        TEXT NOT NULL,
    PRIMARY KEY (owner, entry_id)
)
"""


def new_participant() -> str:
    """A fresh token. Random, not derived from anything about the request.

    Deriving one from an address or a user agent would make it a fingerprint —
    stable across browsers, reconstructible without the cookie, and identifying
    in a way nobody consented to. Random means the only thing that links two
    requests is a cookie the participant can delete.
    """
    from secrets import token_urlsafe

    return "p-" + token_urlsafe(12)


def participant_in(request) -> Optional[str]:
    """The token this request carries, if it carries one."""
    found = request.cookies.get(COOKIE)
    return found if found else None


def attach(response, participant: str) -> None:
    """Put the token on the response.

    `httponly` so page scripts cannot read it, `samesite=lax` so it is not sent
    from another site's form post, `secure` under production so it is not sent
    in clear. None of these protects a secret — the token is not one — but a
    value that identifies a study participant should not be casually readable
    or casually replayable either.
    """
    from ..deploy.context import current

    try:
        secure = current().is_production
    except Exception:                                          # noqa: BLE001
        secure = False
    response.set_cookie(COOKIE, participant, max_age=60 * 60 * 24 * 30,
                        httponly=True, samesite="lax", path="/", secure=secure)


#: Pilot surfaces. Everything else under `/workspace` is the old product.
PILOT_PATHS = ("/workspace/new", "/pilot")


def is_a_departure(path: str) -> bool:
    """Whether this path is the legacy workspace rather than the pilot.

    Written as a predicate over the path so it can be tested without a request,
    and listed positively — the pilot surfaces are two and the legacy ones are
    dozens, so enumerating the legacy side would go stale the first time
    somebody adds a route.
    """
    if any(path == one or path.startswith(one + "/") for one in PILOT_PATHS):
        return False
    return path == "/workspace" or path.startswith("/workspace/")


def note_departures(app) -> None:
    """Record a pilot participant reaching the legacy workspace.

    Middleware rather than a call in the legacy handlers. Those handlers
    predate the pilot, they serve people who are not in it, and editing forty
    of them to ask "is this a study participant" would put the experiment's
    concerns inside the product's — which is the coupling the whole pilot seam
    exists to avoid.

    Only under a runtime deployment, and only for a request already carrying a
    token: a first-time visitor landing on `/workspace/` has not *left*
    anything, and counting them would turn ordinary traffic into evidence of
    abandonment.
    """
    from starlette.middleware.base import BaseHTTPMiddleware

    async def watch(request, call_next):
        response = await call_next(request)
        try:
            from ..deploy.context import ParserMode, current
            from .pilot_events import observe_departure

            participant = participant_in(request)
            if (participant
                    and current().model.mode is ParserMode.RUNTIME
                    and is_a_departure(request.url.path)):
                route = request.scope.get("route")
                observe_departure(getattr(route, "path", request.url.path),
                                  participant=participant)
        except Exception:                                      # noqa: BLE001
            pass
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=watch)


def _connect():
    from ..db.engine import Database
    from ..deploy.context import current

    from .study_repair import ensure_owner

    connection = Database(current().database.url).connect()
    connection.execute(SCHEMA)
    ensure_owner(connection, "pilot_transcripts")
    return connection


def retained() -> bool:
    """Whether this deployment keeps prose. Asked, never assumed."""
    from ..deploy.context import current

    return current().study.retain_transcripts


def last_prompt(participant: str) -> str:
    """The previous sentence, for deciding whether this one is a revision.

    Returns "" when nothing is retained, which is why the caller must treat an
    empty previous prompt as "unknown" rather than as "they typed nothing" —
    a deployment with transcripts off still counts resubmissions, it just
    cannot say whether the wording changed.
    """
    if not participant:
        return ""
    try:
        connection = _connect()
    except Exception:                                          # noqa: BLE001
        return ""
    try:
        from .owner import current as owner_of

        row = connection.execute(
            "SELECT text FROM pilot_transcripts "
            "WHERE owner = ? AND participant = ? "
            "ORDER BY at DESC LIMIT 1",
            (owner_of(), participant)).fetchone()
        return str(row["text"]) if row else ""
    except Exception:                                          # noqa: BLE001
        return ""
    finally:
        connection.close()


def record(participant: str, text: str, attempt: int, **detail: Any) -> None:
    """One attempt, verbatim, if this deployment *and* this person said so.

    Two gates, not one. `retained()` is the deployment's declaration that a
    study is running at all; `may_keep_prose` is whether this particular
    participant agreed. A deployment switch alone would keep the words of the
    one person who declined along with the nine who did not, which is the
    opposite of what the notice promises them.

    `attempt` is passed in rather than counted here, and it comes from the
    events table. A transcript-local counter would number the same submission
    differently whenever transcripts had been off for a while, so a chain read
    beside the counts would show attempt 2 next to attempt 5 for one action.
    One source for the number, and it is the one that works in every
    deployment.

    Never raises into a request, for the same reason the event recorder does
    not: a participant losing their plan because a study table was locked would
    be a worse outcome than losing the transcript.
    """
    from .pilot_consent import may_keep_prose

    if not may_keep_prose(participant):
        return

    from datetime import datetime, timezone
    from hashlib import sha256

    at = datetime.now(timezone.utc).isoformat()
    entry_id = sha256(f"{participant}{at}{text}".encode()).hexdigest()[:16]
    try:
        connection = _connect()
        try:
            from .owner import current as owner_of

            connection.execute(
                "INSERT INTO pilot_transcripts "
                "(owner, entry_id, participant, at, attempt, text, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_of(), entry_id, participant, at, attempt, text,
                 json.dumps(detail, default=str)))
            connection.commit()
        finally:
            connection.close()
    except Exception:                                          # noqa: BLE001
        return


def transcript(participant: str) -> Sequence[Mapping[str, Any]]:
    """One participant's chain, in order. For reading before their interview."""
    connection = _connect()
    try:
        from .owner import current as owner_of

        rows = connection.execute(
            "SELECT at, attempt, text, detail FROM pilot_transcripts "
            "WHERE owner = ? AND participant = ? ORDER BY attempt",
            (owner_of(), participant)).fetchall()
    finally:
        connection.close()
    return [{"at": r["at"], "attempt": r["attempt"], "text": r["text"],
             **json.loads(r["detail"])} for r in rows]


def every_participant() -> Sequence[str]:
    connection = _connect()
    try:
        from .owner import current as owner_of

        rows = connection.execute(
            "SELECT DISTINCT participant FROM pilot_transcripts "
            "WHERE owner = ?", (owner_of(),)).fetchall()
    finally:
        connection.close()
    # A cleared pseudonym is not a participant. `participant` is nullable so it
    # can be scrubbed while the words stay, and a NULL reaching this list would
    # appear as an anonymous subject who never existed.
    return [r["participant"] for r in rows if r["participant"]]


def expire(now=None) -> int:
    """Drop transcripts past the declared retention. Returns how many went.

    A retention setting nothing enforces is a promise, and this pilot will make
    that promise to about ten people in person.
    """
    from datetime import datetime, timedelta, timezone

    from ..deploy.context import current

    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=current().study.retention_days)).isoformat()
    connection = _connect()
    try:
        from .owner import current as owner_of

        # Aliased and read by name. `removed[0]` is a lookup for a key named 0
        # where a row is only a mapping — correct on one dialect, a `KeyError`
        # on the other — and this one sits inside retention, where the
        # exception would be swallowed and the deployment would report having
        # deleted nothing while deleting everything.
        owner = owner_of()
        removed = connection.execute(
            "SELECT COUNT(*) AS how_many FROM pilot_transcripts "
            "WHERE owner = ? AND at < ?", (owner, cutoff)).fetchone()
        connection.execute(
            "DELETE FROM pilot_transcripts WHERE owner = ? AND at < ?",
            (owner, cutoff))
        connection.commit()
        return int(removed["how_many"]) if removed else 0
    finally:
        connection.close()


def forget(participant: str) -> int:
    """Everything one participant typed, gone. Returns how many rows.

    Someone who asks to be removed from a ten-person study is asking a person,
    not a form, so this is a function to call at a prompt — but it has to exist
    before the asking, or the answer is "we would have to write something".
    """
    connection = _connect()
    try:
        from .owner import current as owner_of

        # The same positional read, and the same fix. This one answers somebody
        # who asked to be removed from the study, so a swallowed error here
        # would report a deletion that did not happen.
        owner = owner_of()
        removed = connection.execute(
            "SELECT COUNT(*) AS how_many FROM pilot_transcripts "
            "WHERE owner = ? AND participant = ?",
            (owner, participant)).fetchone()
        connection.execute(
            "DELETE FROM pilot_transcripts WHERE owner = ? AND participant = ?",
            (owner, participant))
        connection.commit()
        return int(removed["how_many"]) if removed else 0
    finally:
        connection.close()
