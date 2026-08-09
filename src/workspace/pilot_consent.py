"""Whether one particular person agreed, and to which words.

The deployment switch (`QUANTIFY_PILOT_TRANSCRIPTS`) says the study *may* keep
prose. It cannot say whether a given participant agreed, and a notice reading
"transcript recording is disabled unless you explicitly agree" is a promise
about a person, not about a deployment. One switch for ten people would record
the one who declined along with the nine who did not.

So consent is per participant, and it fails closed:

    UNKNOWN     nobody has been asked yet         no prose kept
    DECLINED    they were asked and said no       no prose kept
    GRANTED     they were asked and agreed        prose kept

`UNKNOWN` and `DECLINED` behave identically and are still stored apart, because
an empty transcript store otherwise has two very different meanings — nobody
was asked, or everybody refused — and those lead to opposite conclusions about
whether the study was run properly. That is the same ambiguity as zero events
versus zero usage.

**Consent is not retroactive.** Prompts typed before someone agreed stay
unkept. There is no path here that reaches back and retains what was already
discarded, which is a property rather than an omission: agreeing at the end of
a session is agreeing to what happens next, and a system that swept up the
earlier prompts would be keeping words the person had not been asked about when
they typed them.

**The wording is versioned.** Somebody who agreed to one notice has not agreed
to a later, broader one. A changed `NOTICE` bumps `NOTICE_VERSION`, and an
earlier grant stops counting until they are asked again.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

UNKNOWN = "UNKNOWN"
DECLINED = "DECLINED"
GRANTED = "GRANTED"

#: Bump whenever `NOTICE` changes in substance. Consent recorded against an
#: older version stops counting — see `state_of`.
NOTICE_VERSION = "2026-08-09.1"

#: The exact words a participant agrees to, kept in the repository so that what
#: was shown and what was recorded cannot drift apart. A consent record naming
#: a version whose text nobody can produce is not evidence of consent.
NOTICE = """\
Pilot notice

During this pilot we may record the prompts you enter and the system's
responses to help us improve the interpreter. These transcripts are used only
for product improvement, are retained for up to 30 days, and can be deleted on
request at any time. Participation is voluntary, and transcript recording is
disabled unless you explicitly agree.\
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_consent (
    participant     TEXT PRIMARY KEY,
    state           TEXT NOT NULL,
    at              TEXT NOT NULL,
    notice_version  TEXT NOT NULL
)
"""


def _connect():
    from ..db.engine import Database
    from ..deploy.context import current

    connection = Database(current().database.url).connect()
    connection.execute(SCHEMA)
    return connection


def _set(participant: str, state: str) -> None:
    from datetime import datetime, timezone

    connection = _connect()
    try:
        connection.execute("DELETE FROM pilot_consent WHERE participant = ?",
                           (participant,))
        connection.execute(
            "INSERT INTO pilot_consent (participant, state, at, notice_version)"
            " VALUES (?, ?, ?, ?)",
            (participant, state, datetime.now(timezone.utc).isoformat(),
             NOTICE_VERSION))
        connection.commit()
    finally:
        connection.close()


def grant(participant: str) -> None:
    """Record that this person agreed, to the notice as it currently reads.

    Called by whoever is running the session, after reading the notice out.
    Not a checkbox on the page: this pilot is five to ten people in a
    conversation, and a checkbox would collect a click where the protocol
    wants somebody to have actually said it.
    """
    _set(participant, GRANTED)


def decline(participant: str) -> None:
    """Record that this person was asked and said no.

    Distinct from never asking. Recorded so the study can tell "we did not run
    the protocol" from "we ran it and they declined" — and so nobody is asked
    twice in one session.
    """
    _set(participant, DECLINED)


def withdraw(participant: str) -> int:
    """They changed their mind. Consent is revoked and the words are deleted.

    One call, because "you can have it deleted on request at any time" is a
    single promise and splitting it across two functions is how half of it gets
    kept. Returns how many transcript rows went.
    """
    from .pilot_session import forget

    _set(participant, DECLINED)
    return forget(participant)


def record_of(participant: str) -> Optional[Mapping[str, str]]:
    """The stored decision, whatever version it was made against."""
    if not participant:
        return None
    try:
        connection = _connect()
    except Exception:                                          # noqa: BLE001
        return None
    try:
        row = connection.execute(
            "SELECT state, at, notice_version FROM pilot_consent "
            "WHERE participant = ?", (participant,)).fetchone()
    except Exception:                                          # noqa: BLE001
        return None
    finally:
        connection.close()
    if row is None:
        return None
    return {"state": row[0], "at": row[1], "notice_version": row[2]}


def state_of(participant: str) -> str:
    """What this participant has agreed to *now*.

    A grant against a superseded notice reads as `UNKNOWN`, not as `GRANTED`.
    Somebody who agreed to thirty-day retention has not agreed to whatever a
    later notice says, and treating an old yes as a current one is how a study
    ends up holding words under terms nobody accepted.
    """
    stored = record_of(participant)
    if stored is None:
        return UNKNOWN
    if stored["state"] == GRANTED and stored["notice_version"] != NOTICE_VERSION:
        return UNKNOWN
    return stored["state"]


def may_keep_prose(participant: str) -> bool:
    """The one question `pilot_session.record` asks. Both gates, in order.

    The deployment gate first: an instance that never turned the study on keeps
    nothing regardless of what anybody agreed to, so a developer checkout
    cannot accumulate transcripts because a consent row exists in a copied
    database.
    """
    from .pilot_session import retained

    # `==`, not `is`. `state_of` returns a string loaded from the database,
    # which is never the same object as the module constant even when it is the
    # same text — an identity test here would have refused every real grant
    # while passing any test that stubbed the function.
    return bool(participant) and retained() and state_of(participant) == GRANTED


def by_state() -> Mapping[str, Sequence[str]]:
    """Everyone the study has a decision for, grouped.

    For the operator, so "no transcripts" is never ambiguous: it reads as eight
    declines or as eight people nobody asked, and those say different things
    about whether the protocol was followed.
    """
    from .pilot_events import every_event

    seen = {e.get("participant") for e in every_event() if e.get("participant")}
    grouped: dict = {UNKNOWN: [], DECLINED: [], GRANTED: []}
    for participant in sorted(seen):
        grouped.setdefault(state_of(participant), []).append(participant)
    return grouped
