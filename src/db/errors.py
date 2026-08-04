"""Database failures as operator evidence, never as API payloads.

    driver exception -> classify(SQLSTATE) -> DatabaseFailure -> envelope
                     \\_ preserved on __cause__ for the operator log

A driver exception is a description of the deployment. `psycopg` reports a
foreign-key violation as:

    foreign key constraint "fk_plan_run_access_event" on table "plan_run"
    DETAIL: Key (owner, access_event_id)=(alice, mdae-3f1c) is still referenced

One string: a constraint name, two table names, the column composition of a key,
a tenant identifier and one of that tenant's object ids. Anything that puts it in
a response has published the schema and told the caller that `alice` exists and
owns something.

**The public category is semantic; the internal reason is diagnostic.** A
missing parent and a cross-tenant reference are both `CONSTRAINT_CONFLICT` to a
caller — distinguishing them publicly is exactly what would leak another
tenant's existence — while operators and metrics keep `MISSING_PARENT` and
`CROSS_SCOPE_REFERENCE` apart, because the first is a client ordering mistake
and the second is an authorization boundary being probed.

**Classified by SQLSTATE, not by exception class.** The codes are standard and
stable across drivers and versions; matching `psycopg.errors.DeadlockDetected`
would tie this to one driver, would miss a subclass, and would put `psycopg`
into modules that only need to know someone else got there first.

**Retryability is decided here, not by the route.** A caller needs to know
whether to retry unchanged, re-read and re-plan, or stop. That is a property of
what went wrong; a route inferring it from a status code would be
reconstructing what this layer already knew.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PublicCode(str, Enum):
    """What a caller is told. Small, semantic, free of deployment facts."""

    DATABASE_CONTENTION = "DATABASE_CONTENTION"
    CONSTRAINT_CONFLICT = "CONSTRAINT_CONFLICT"
    STALE_TRANSITION = "STALE_TRANSITION"
    TRANSITION_INTEGRITY_FAILURE = "TRANSITION_INTEGRITY_FAILURE"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    DATABASE_INTERNAL_FAILURE = "DATABASE_INTERNAL_FAILURE"


class InternalReason(str, Enum):
    """What an operator is told. Never serialized to a client.

    These exist because "constraint conflict" is not actionable at three in the
    morning. A missing parent is a client ordering mistake; a cross-scope
    reference is someone reaching across a tenant boundary. They deserve
    different alerts and the same public answer.
    """

    SERIALIZATION_FAILURE = "SERIALIZATION_FAILURE"
    DEADLOCK = "DEADLOCK"
    LOCK_UNAVAILABLE = "LOCK_UNAVAILABLE"

    DUPLICATE_IDENTITY = "DUPLICATE_IDENTITY"
    MISSING_PARENT = "MISSING_PARENT"
    CROSS_SCOPE_REFERENCE = "CROSS_SCOPE_REFERENCE"
    DEPENDENT_RECORD_EXISTS = "DEPENDENT_RECORD_EXISTS"
    CHECK_VIOLATION = "CHECK_VIOLATION"
    EXCLUSION_VIOLATION = "EXCLUSION_VIOLATION"
    LEGACY_INTEGRITY_FAILURE = "LEGACY_INTEGRITY_FAILURE"

    NO_ROW_TRANSITIONED = "NO_ROW_TRANSITIONED"
    TOO_MANY_ROWS_TRANSITIONED = "TOO_MANY_ROWS_TRANSITIONED"

    UNREACHABLE = "UNREACHABLE"
    ADMIN_SHUTDOWN = "ADMIN_SHUTDOWN"
    UNCLASSIFIED = "UNCLASSIFIED"


class Retry(str, Enum):
    """What the caller should do, decided from what went wrong."""

    UNCHANGED = "UNCHANGED"
    """Reissue the same request after a delay. Nothing about it was wrong."""

    AFTER_REREAD = "AFTER_REREAD"
    """Re-read the authoritative state and re-plan first. Retrying unchanged
    would repeat a decision made against state that has since moved."""

    NEVER = "NEVER"
    """The request cannot succeed as issued, or something is wrong that a
    retry would only repeat."""

    @property
    def retryable(self) -> bool:
        return self is not Retry.NEVER


#: SQLSTATE -> (public code, internal reason, what to do).
#:
#: `23503` deliberately does not sit under the old `is_conflict` boolean. A
#: foreign-key violation and a duplicate key are both `CONSTRAINT_CONFLICT`
#: publicly and are not the same event: one means a referenced record is absent
#: or out of scope, the other that this identity is already taken.
_BY_SQLSTATE = {
    "40001": (PublicCode.DATABASE_CONTENTION, InternalReason.SERIALIZATION_FAILURE,
              Retry.UNCHANGED),
    "40P01": (PublicCode.DATABASE_CONTENTION, InternalReason.DEADLOCK,
              Retry.UNCHANGED),
    "55P03": (PublicCode.DATABASE_CONTENTION, InternalReason.LOCK_UNAVAILABLE,
              Retry.UNCHANGED),

    "23505": (PublicCode.CONSTRAINT_CONFLICT, InternalReason.DUPLICATE_IDENTITY,
              Retry.AFTER_REREAD),
    "23503": (PublicCode.CONSTRAINT_CONFLICT, InternalReason.MISSING_PARENT,
              Retry.AFTER_REREAD),
    "23514": (PublicCode.CONSTRAINT_CONFLICT, InternalReason.CHECK_VIOLATION,
              Retry.NEVER),
    "23P01": (PublicCode.CONSTRAINT_CONFLICT, InternalReason.EXCLUSION_VIOLATION,
              Retry.AFTER_REREAD),

    "08000": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
    "08001": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
    "08003": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
    "08004": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
    "08006": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
    "08007": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.AFTER_REREAD),
    "57P01": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.ADMIN_SHUTDOWN,
              Retry.UNCHANGED),
    "57P03": (PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
              Retry.UNCHANGED),
}

#: The public sentence for each code. Fixed strings, never interpolated — a
#: message built from an exception carries whatever that exception carried.
_MESSAGES = {
    PublicCode.DATABASE_CONTENTION:
        "The service was busy with another change to the same records. "
        "Nothing from this attempt was kept.",
    PublicCode.CONSTRAINT_CONFLICT:
        "The requested change conflicts with the current stored state.",
    PublicCode.STALE_TRANSITION:
        "The record changed after it was read. Read it again before retrying.",
    PublicCode.TRANSITION_INTEGRITY_FAILURE:
        "The change could not be applied safely and was abandoned.",
    PublicCode.DATABASE_UNAVAILABLE:
        "The service is temporarily unable to reach its storage.",
    PublicCode.DATABASE_INTERNAL_FAILURE:
        "The request could not be completed.",
}

#: Which HTTP status each code becomes.
_STATUS = {
    PublicCode.CONSTRAINT_CONFLICT: 409,
    PublicCode.STALE_TRANSITION: 409,
    PublicCode.TRANSITION_INTEGRITY_FAILURE: 500,
    PublicCode.DATABASE_UNAVAILABLE: 503,
    PublicCode.DATABASE_INTERNAL_FAILURE: 500,
}


@dataclass(frozen=True)
class Classification:
    code: PublicCode
    reason: InternalReason
    retry: Retry
    sqlstate: Optional[str] = None


class DatabaseFailure(Exception):
    """A database failure, safe to let reach any boundary.

    Carries the public category and the private reason on one object, so a
    caller cannot serialize the second by reaching for "the error". The driver
    exception stays on `__cause__` — raised `from exc`, never `from None`,
    because sanitising the public channel must not blind the operator one.
    """

    def __init__(self, classification: Classification, *, operation: str = "",
                 request_id: str = "") -> None:
        self.classification = classification
        self.operation = operation
        """What was being attempted, in application terms. Private."""

        self.request_id = request_id
        super().__init__(self.public_message)

    @property
    def code(self) -> PublicCode:
        return self.classification.code

    @property
    def reason(self) -> InternalReason:
        return self.classification.reason

    @property
    def retry(self) -> Retry:
        return self.classification.retry

    @property
    def status(self) -> int:
        """Contention splits on what the caller must do rather than on the
        category: `503` when the same request may be reissued unchanged, `409`
        when the state it was planned against has moved. The route does not
        make that judgment — the failure already knows."""
        if self.code is PublicCode.DATABASE_CONTENTION:
            return 503 if self.retry is Retry.UNCHANGED else 409
        return _STATUS[self.code]

    @property
    def public_message(self) -> str:
        return _MESSAGES[self.code]

    def public(self) -> dict:
        """The whole client-visible payload. Nothing else may be added.

        Built from fixed strings and enum values only. There is no path from
        the driver exception into this dictionary, which is what makes the leak
        structurally impossible rather than merely avoided — `str(self)` is the
        fixed message, so even a careless `str(exc)` at a boundary is safe.
        """
        payload = {"code": self.code.value, "message": self.public_message,
                   "retryable": self.retry.retryable}
        if self.request_id:
            payload["request_id"] = self.request_id
        return payload

    def private(self) -> dict:
        """What the operator log keeps. Never serialized to a client."""
        cause = self.__cause__
        return {"code": self.code.value, "reason": self.reason.value,
                "retry": self.retry.value,
                "sqlstate": self.classification.sqlstate,
                "operation": self.operation, "request_id": self.request_id,
                "driver_exception": type(cause).__name__ if cause else None,
                "driver_detail": str(cause) if cause else ""}


def sqlstate(error: BaseException) -> Optional[str]:
    """The SQLSTATE an exception carries, if it carries one."""
    code = getattr(error, "sqlstate", None)
    if code:
        return str(code)
    diagnostic = getattr(error, "diag", None)
    code = getattr(diagnostic, "sqlstate", None) if diagnostic else None
    return str(code) if code else None


#: SQLite carries no SQLSTATE at all, so its messages are matched — narrowly,
#: and only for the phrases it actually uses. Order matters: "foreign key" is
#: tested before the generic phrases so it is not swallowed by them.
_SQLITE_PHRASES = (
    ("database is locked", "40P01"),
    ("database table is locked", "40P01"),
    ("foreign key constraint failed", "23503"),
    ("unique constraint failed", "23505"),
    ("check constraint failed", "23514"),
    ("not null constraint failed", "23514"),
    ("unable to open database", "08006"),
)


def classify(error: BaseException) -> Classification:
    """What kind of failure this is, from the code rather than the class.

    Unclassified is `DATABASE_INTERNAL_FAILURE` and not retryable. Failing
    towards "stop" is the safe direction: a retry loop on an unknown fault
    turns one failure into sustained load against a database already unwell.
    """
    code = sqlstate(error)
    if code is None:
        text = str(error).lower()
        for phrase, mapped in _SQLITE_PHRASES:
            if phrase in text:
                code = mapped
                break

    if code in _BY_SQLSTATE:
        public, reason, retry = _BY_SQLSTATE[code]
        return Classification(public, reason, retry, code)

    if code and code.startswith("08"):
        # The whole connection-exception class, so a code not enumerated above
        # is still recognised as "the database is not reachable" rather than
        # becoming an unclassified internal fault.
        return Classification(PublicCode.DATABASE_UNAVAILABLE,
                              InternalReason.UNREACHABLE, Retry.UNCHANGED, code)

    return Classification(PublicCode.DATABASE_INTERNAL_FAILURE,
                          InternalReason.UNCLASSIFIED, Retry.NEVER, code)


def translate(error: BaseException, *, operation: str = "",
              request_id: str = "",
              reason: Optional[InternalReason] = None) -> DatabaseFailure:
    """Wrap a driver exception, keeping it on `__cause__`.

    `reason` narrows the internal classification where the caller knows more
    than the SQLSTATE does — a `23503` is `MISSING_PARENT` in general, and
    `CROSS_SCOPE_REFERENCE` when the writer knows the parent exists under a
    different owner. The public code does not change: telling a caller which of
    the two it was is telling them another tenant holds that id.
    """
    classification = classify(error)
    if reason is not None:
        classification = Classification(classification.code, reason,
                                        classification.retry,
                                        classification.sqlstate)
    failure = DatabaseFailure(classification, operation=operation,
                              request_id=request_id)
    failure.__cause__ = error
    return failure


#: Codes that mean "retry or report a conflict", not "your request was wrong".
CONFLICT_SQLSTATES = frozenset({"40001", "40P01", "23505", "55P03"})


def is_conflict(error: BaseException) -> bool:
    """Whether this exception is two sessions contending.

    Kept for the apply path, which reports contention specifically. Narrower
    than `classify` on purpose: a foreign-key violation is a conflict with
    stored state and is not two sessions racing, and folding it in here would
    make `accept` report "another request was changing the same records" for a
    plan that simply is not there.

    Recognises an already-translated failure as well as a raw driver one. The
    engine now classifies before this is reached, so the version that only
    inspected SQLSTATE stopped seeing anything: contention arrived as a
    `DatabaseFailure`, `is_conflict` said no, and the apply path lost the
    domain refusal it exists to produce. That is the general hazard of adding
    a translation layer beneath code that was reading raw errors — the layer
    is correct and every consumer above it is now looking for the wrong shape.
    """
    if isinstance(error, DatabaseFailure):
        return error.code is PublicCode.DATABASE_CONTENTION
    code = sqlstate(error)
    if code is not None:
        return code in CONFLICT_SQLSTATES
    text = str(error).lower()
    return "database is locked" in text or "database table is locked" in text
