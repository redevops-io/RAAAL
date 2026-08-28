"""The anonymous evaluation session (§4 of the public strategy-lab plan).

An explicit, ephemeral wrapper around an *already-evaluated* artifact. It exists
so that a visitor who evaluated a strategy without an account can save the exact
result after logging in — and so that "save" is a binding of the artifact to an
owner rather than a fresh interpretation of the sentence.

**What it is not.** It is not a durable private workspace for anonymous users
(§4 forbids that). The evaluated artifact it points at is the content-addressed
review already persisted in `pilot_reviews`; the session itself is a short-TTL
envelope carrying only *references* to that artifact and its provenance, plus a
single-use save token. It holds no user identity — the whole point is that the
owner is decided at save time, after authentication, and is never part of what
the session is.

**The critical invariant it protects (§2/§4).** After login, the save binds the
exact evaluated artifact — no provider/model call is required merely to save an
already-evaluated plan. The session records `compiled_plan_hash` — the review's
content-addressed id — and `evaluated_plan_id`, the plan identity that the
evaluated artifact *already determines*. Completing the save reopens that review
(a dict-only operation that cannot construct a reader) and asserts the minted
plan id equals `evaluated_plan_id`. A model call on the save path could only
change that hash, so it cannot occur unnoticed.

**Where it lives.** A process-local, short-TTL store keyed by the opaque
`session_id`, not a database table. Anonymous evaluation must not create durable
per-user rows, and the durable half — the review and, after login, the plan —
already lives in the content-addressed `pilot_reviews`/`pilot_plans` stores. The
session store holds nothing whose loss matters: a lapsed session is re-created
from the still-persisted review the next time the visitor clicks Save. Making it
process-local is what keeps "anonymous evaluation is not a durable workspace" a
property of the code rather than a promise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from typing import Any, Dict, Mapping, Optional, Sequence

from .owner import SHARED

#: How long an evaluation session is resolvable. Short: it is a bridge across a
#: login round-trip, not a place to keep anything. A visitor who signs in and
#: comes back within the hour completes their save; one who does not re-clicks
#: Save and gets a fresh session over the same (still-persisted) review.
DEFAULT_TTL_SECONDS = 60 * 60


class SessionError(RuntimeError):
    """A session could not be created, resolved, or consumed as asked."""


@dataclass(frozen=True)
class EvaluationSession:
    """One anonymous evaluation, wrapped so its result can be bound to an owner.

    Every field is either a reference to the already-persisted evaluated
    artifact or impersonal provenance. There is deliberately no `owner`,
    `subject`, `email`, credential, or account field: an anonymous session that
    named a user would be the durable private workspace §4 rules out, and the
    owner is a property of the *save*, decided after authentication.
    """

    #: Opaque, random, unguessable. The address of this session and nothing else
    #: — it carries no information about the visitor.
    session_id: str

    #: The words the visitor typed. Strategy meaning, not account state.
    original_prompt: str

    #: A reference to the parsed intent (its hash), never the sentence re-parsed.
    parsed_intent: str

    #: The visitor's own answers to the interpreter's questions, by dimension.
    clarification_answers: Mapping[str, str]

    #: The evaluated artifact's content address — the `pilot_reviews` review id.
    #: This *is* the identity of the thing being saved; the owner is the envelope
    #: around it, not part of it.
    compiled_plan_hash: str

    #: The plan identity the evaluated artifact already determines, computed at
    #: create time by reopening the review (no reader). The save re-derives this
    #: and must match — the structural proof that nothing was recomputed.
    evaluated_plan_id: str

    #: Which catalogue entry, if any, supplied the assumptions under the answers.
    #: Part of the artifact's identity (a review of the same words against a
    #: different pick is a different review), so it travels with the session.
    picked: str

    #: The owner scope the review was persisted under at evaluation time — the
    #: shared anonymous workspace for a signed-out visitor. The save reads the
    #: review from here and writes the plan under the *now-authenticated* owner.
    review_owner: str

    #: Impersonal provenance refs (§4). References to published artifacts, not
    #: anything account-specific.
    evaluation_artifact_ids: Sequence[str]
    methodology_id: str
    protocol_version: str
    market_data_snapshot_id: str

    created_at: str
    expires_at: str

    #: Single-use. Opaque, random, and bound to this session (a token that
    #: matches no live session, or a consumed one, is rejected). Consuming it is
    #: what makes a replayed save a no-op rather than a second plan.
    save_token: str

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= _parse(self.expires_at)

    def as_dict(self) -> Dict[str, Any]:
        """The envelope, for logging or a signed cookie. Contains no secret.

        The `save_token` is deliberately excluded: this is the shape that is safe
        to render or persist somewhere a viewer can see, and the token is the one
        value that authorises the save.
        """
        return {
            "session_id": self.session_id,
            "original_prompt": self.original_prompt,
            "parsed_intent": self.parsed_intent,
            "clarification_answers": dict(self.clarification_answers),
            "compiled_plan_hash": self.compiled_plan_hash,
            "evaluated_plan_id": self.evaluated_plan_id,
            "picked": self.picked,
            "review_owner": self.review_owner,
            "evaluation_artifact_ids": list(self.evaluation_artifact_ids),
            "methodology_id": self.methodology_id,
            "protocol_version": self.protocol_version,
            "market_data_snapshot_id": self.market_data_snapshot_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


# --- the process-local, short-TTL store ------------------------------------
#
# Not a database table. See the module docstring: the durable artifacts already
# live in the content-addressed stores, and anonymous evaluation must not mint a
# durable per-user row. Each record is the frozen session plus a `consumed` flag;
# a token index lets `consume_save_token` accept the token alone, as §4 asks.

@dataclass
class _Record:
    session: EvaluationSession
    consumed: bool = False


_SESSIONS: Dict[str, _Record] = {}
_BY_TOKEN: Dict[str, str] = {}


def _sweep(now: Optional[datetime] = None) -> None:
    """Drop expired sessions. A TTL nothing enforces is not a TTL."""
    now = now or datetime.now(timezone.utc)
    dead = [sid for sid, rec in _SESSIONS.items()
            if rec.session.is_expired(now)]
    for sid in dead:
        rec = _SESSIONS.pop(sid, None)
        if rec is not None:
            _BY_TOKEN.pop(rec.session.save_token, None)


def clear() -> None:
    """Forget every session. For tests and for a clean process boundary."""
    _SESSIONS.clear()
    _BY_TOKEN.clear()


def _parse(stamp: str) -> datetime:
    value = datetime.fromisoformat(stamp)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


# --- construction ----------------------------------------------------------

def create_for_review(review_id: str, picked: str = "", *,
                      ttl_seconds: int = DEFAULT_TTL_SECONDS,
                      methodology_id: Optional[str] = None,
                      market_data_snapshot_id: str = "") -> EvaluationSession:
    """Wrap an already-evaluated, already-persisted review as a session.

    The review is content-addressed by `review_id`, so this loads the *exact*
    evaluated artifact rather than re-deriving one — no reader is constructed
    here and none can be. The review is looked for under the current owner and,
    failing that, under the shared anonymous workspace it was written to when the
    visitor was signed out. Whichever owner holds it is remembered as the scope
    the save will read from.

    Raises `SessionError` if the review is gone (a stale link) or carries no
    pinned intent (there is nothing sealed to save).
    """
    from .pilot import reopen
    from .pilot_store import (PILOT_OWNER, load_review, load_review_under,
                              plan_id_for)

    review_owner = PILOT_OWNER()
    stored = load_review(review_id)
    if stored is None:
        stored = load_review_under(SHARED, review_id)
        review_owner = SHARED
    if stored is None:
        raise SessionError(
            f"no evaluated review {review_id!r} to save; the evaluation may "
            "have expired or the link is stale")
    if stored.get("intent") is None:
        raise SessionError(
            "this evaluation has no sealed intent to save — it is still a "
            "question, not a plan")

    # Reopen re-derives the plan identity from the stored artifact. It takes a
    # dict and cannot reach a reader, so `evaluated_plan_id` is the plan the
    # evaluated artifact already determines, computed without interpreting the
    # sentence. The save re-derives it and must match.
    reading = reopen(stored)
    evaluated_plan_id = plan_id_for(reading)

    settled = stored.get("settled", ()) or ()
    clarification_answers = {
        entry.get("field"): entry.get("value")
        for entry in settled
        if entry.get("provenance") == "USER_ANSWERED" and entry.get("field")
    }
    parsed_intent = (stored.get("intent") or {}).get("intent_hash", "")

    now = datetime.now(timezone.utc)
    session = EvaluationSession(
        session_id="es-" + token_urlsafe(18),
        original_prompt=stored.get("text", ""),
        parsed_intent=parsed_intent,
        clarification_answers=clarification_answers,
        compiled_plan_hash=review_id,
        evaluated_plan_id=evaluated_plan_id,
        picked=picked or stored.get("picked", ""),
        review_owner=review_owner,
        evaluation_artifact_ids=(review_id, evaluated_plan_id),
        methodology_id=(methodology_id
                        if methodology_id is not None
                        else stored.get("reader_id", "")),
        protocol_version=stored.get("interpreter_version", ""),
        market_data_snapshot_id=market_data_snapshot_id,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat(),
        save_token="st-" + token_urlsafe(24),
    )
    _SESSIONS[session.session_id] = _Record(session=session)
    _BY_TOKEN[session.save_token] = session.session_id
    return session


def create(reading, picked: str = "", prompt: str = "", *,
           ttl_seconds: int = DEFAULT_TTL_SECONDS,
           methodology_id: Optional[str] = None,
           market_data_snapshot_id: str = "") -> EvaluationSession:
    """Create a session from a live reading, persisting the review first.

    A convenience over `create_for_review` for a caller that holds the reading
    rather than a review id: it persists the content-addressed review (idempotent
    — the same evaluation rewrites the same row) and then wraps it. The review is
    the durable artifact; the session only references it.
    """
    from .pilot_store import review_id_for, save_review

    review_id = save_review(reading, picked)
    # `prompt` is accepted for API symmetry with §4's field list; the persisted
    # review already carries the text, and `create_for_review` reads it there so
    # the two paths cannot disagree about what was typed.
    assert review_id == review_id_for(reading, picked)
    return create_for_review(
        review_id, picked, ttl_seconds=ttl_seconds,
        methodology_id=methodology_id,
        market_data_snapshot_id=market_data_snapshot_id)


# --- resolution, integrity, consumption ------------------------------------

def resolve(session_id: str) -> Optional[EvaluationSession]:
    """The session for this id, or None if it is unknown or expired.

    An expired session is dropped rather than returned: the save it would
    authorise is exactly the one the TTL exists to stop.
    """
    _sweep()
    rec = _SESSIONS.get(session_id)
    if rec is None:
        return None
    if rec.session.is_expired():
        _forget(session_id)
        return None
    return rec.session


def verify(session_id: str, save_token: str) -> bool:
    """Whether this (session_id, save_token) pair is live, matched and unspent.

    Integrity/tamper check: a forged or altered session id resolves to nothing,
    and a save token that does not match the one this session was minted with is
    rejected even if the id is real. Neither is a value a client can invent —
    they are compared against server-held state, not merely re-signed.
    """
    rec = _SESSIONS.get(session_id)
    if rec is None or rec.session.is_expired():
        return False
    if rec.consumed:
        return False
    return _constant_time_eq(save_token, rec.session.save_token)


def consume_save_token(save_token: str) -> Optional[EvaluationSession]:
    """Spend the single-use token, returning its session, or None.

    Single use is the whole point: the first call flips the record to consumed
    and hands back the session so the save can proceed; every later call — a
    double-click, a refresh of the resume URL, a deliberate replay — returns
    None, and no second plan is minted. A token that matches no live session, or
    a session that has expired, also returns None.
    """
    _sweep()
    session_id = _BY_TOKEN.get(save_token)
    if session_id is None:
        return None
    rec = _SESSIONS.get(session_id)
    if rec is None or rec.session.is_expired():
        _forget(session_id)
        return None
    if rec.consumed:
        return None
    if not _constant_time_eq(save_token, rec.session.save_token):
        return None
    rec.consumed = True
    return rec.session


def is_consumed(session_id: str) -> bool:
    rec = _SESSIONS.get(session_id)
    return bool(rec and rec.consumed)


def _forget(session_id: str) -> None:
    rec = _SESSIONS.pop(session_id, None)
    if rec is not None:
        _BY_TOKEN.pop(rec.session.save_token, None)


def _constant_time_eq(a: str, b: str) -> bool:
    from hmac import compare_digest

    try:
        return compare_digest(str(a), str(b))
    except Exception:  # noqa: BLE001
        return False
