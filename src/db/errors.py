"""Recognising a concurrency conflict without knowing which driver raised it.

When two sessions contend, PostgreSQL reports it as a deadlock, a serialization
failure or a unique violation depending on how they collided. Those arrive as
driver exceptions carrying constraint names, table structure, process ids and a
server HINT — none of which belongs in an application error path, and all of
which describes the deployment to whoever sees it.

**Classified by SQLSTATE, not by exception class.** The codes are defined by the
SQL standard and are stable across drivers and versions; matching on
`psycopg.errors.DeadlockDetected` would tie this to one driver and would miss a
subclass. It also keeps `psycopg` out of the modules that merely need to know
"someone else got there first".

**This is a backstop, not the mechanism.** Contention is meant to be settled by
the row lock and the conditional state transition. If one of those is lost, this
is what stops a raw driver error reaching the caller — it was found by removing
the lock in a test and watching `DeadlockDetected` escape.
"""
from __future__ import annotations

from typing import Any, Optional

#: Codes that mean "retry or report a conflict", not "your request was wrong".
#:
#:     40001  serialization_failure
#:     40P01  deadlock_detected
#:     23505  unique_violation
#:     55P03  lock_not_available
CONFLICT_SQLSTATES = frozenset({"40001", "40P01", "23505", "55P03"})


def sqlstate(error: BaseException) -> Optional[str]:
    """The SQLSTATE an exception carries, if it carries one."""
    code = getattr(error, "sqlstate", None)
    if code:
        return str(code)
    diagnostic = getattr(error, "diag", None)
    code = getattr(diagnostic, "sqlstate", None) if diagnostic else None
    return str(code) if code else None


def is_conflict(error: BaseException) -> bool:
    """Whether this exception is two sessions contending.

    SQLite raises a plain `OperationalError` for a busy database with no
    SQLSTATE at all, so its message is checked as well — narrowly, and only for
    the two phrases it actually uses.
    """
    code = sqlstate(error)
    if code is not None:
        return code in CONFLICT_SQLSTATES
    text = str(error).lower()
    return "database is locked" in text or "database table is locked" in text
