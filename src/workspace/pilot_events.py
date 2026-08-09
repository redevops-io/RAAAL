"""What the pilot can count, and deliberately nothing else.

Five events, each an objective fact about what happened:

    plan_compiled       a sentence became a plan
    plan_saved          a plan was persisted
    plan_reopened       a saved plan was opened again
    plan_result_shown   the plan produced a figure
    plan_refused        it did not, or a capability was refused by name

**Structured context only, and never raw user text.** `refusal_code` and
`capability` are named reasons the engine gave; `open_question_count` is a
number. The first version stored the `unavailable` *message* — rendered copy,
which can embed whatever the user typed, and which changes whenever the wording
does. A telemetry field that moves when a sentence is reworded is a field
nobody can count across a cohort.

Every event carries `parser_mode` and `pilot_reader`, so a model-only cohort and
a future dual-witness one can be separated later. Without that, the two would be
one undifferentiated population by the time anyone thought to ask.

**What is not here, and will not be.** Trust, understanding, confusion,
satisfaction. Those are interview questions, and a telemetry field named
`understood_refusal` would be a number derived from nothing — the
classification-without-evidence defect at the point where it would do the most
damage, in the data used to decide whether the runtime was worth building.

A dashboard reporting "82% understood the refusal" is worse than no dashboard,
because somebody will believe it. What this module can honestly say is "of 40
plans, 12 were refused for `assets` and 3 of those were never saved" — and the
question of whether those three people understood why is asked of them.

**Read-only for analysis.** No route exposes these; `summary()` is for whoever
is running the pilot, at a prompt.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_events (
    event_id      TEXT PRIMARY KEY,
    at            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    plan_id       TEXT,
    detail        TEXT NOT NULL
)
"""

PLAN_COMPILED = "plan_compiled"
PLAN_SAVED = "plan_saved"
PLAN_REOPENED = "plan_reopened"
PLAN_RESULT_SHOWN = "plan_result_shown"
PLAN_REFUSED = "plan_refused"

KINDS = (PLAN_COMPILED, PLAN_SAVED, PLAN_REOPENED, PLAN_RESULT_SHOWN,
         PLAN_REFUSED)

#: Coarse, stable reasons an execution produced no figure. Codes rather than
#: the engine's message: a message is rendered copy, it can carry the user's
#: own words, and it changes whenever the wording does.
NO_MARKET_DATA = "NO_MARKET_DATA"
COVERAGE_REFUSED = "COVERAGE_REFUSED"
ENGINE_REFUSED = "ENGINE_REFUSED"
CAPABILITY_REFUSED = "CAPABILITY_REFUSED"


def _execution_refusal_code(run) -> str:
    """Which kind of refusal this was, without reading the message."""
    coverage = run.get("coverage")
    if coverage is not None and not getattr(coverage, "publishable", True):
        return COVERAGE_REFUSED
    if "market data is not available" in str(run.get("unavailable") or ""):
        return NO_MARKET_DATA
    return ENGINE_REFUSED


def _connect():
    from ..deploy.context import current
    from ..db.engine import Database

    connection = Database(current().database.url).connect()
    connection.execute(SCHEMA)
    return connection


def _profile() -> dict:
    """The deployment, on every event.

    Recorded per event rather than assumed constant for a run: a pilot that
    changed profile halfway would otherwise have one population wearing two
    labels, and nothing in the data would say when it happened.
    """
    from ..deploy.context import current

    model = current().model
    return {"parser_mode": model.mode.value,
            "pilot_reader": model.pilot_reader.value}


def record(kind: str, *, plan_id: str = "", **detail: Any) -> None:
    """One event. Never raises into a request.

    Telemetry is the expendable half: a pilot user losing their plan because an
    analytics table was locked would be a worse outcome than losing the count.
    The same rule `deploy.context` already applies to trace retention.
    """
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not one of {KINDS}")

    from datetime import datetime, timezone
    from hashlib import sha256

    at = datetime.now(timezone.utc).isoformat()
    payload = {**detail, **_profile()}
    event_id = sha256(f"{at}{kind}{plan_id}{json.dumps(payload, sort_keys=True)}"
                      .encode()).hexdigest()[:16]
    try:
        connection = _connect()
        try:
            connection.execute(
                "INSERT INTO pilot_events (event_id, at, kind, plan_id, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, at, kind, plan_id, json.dumps(payload)))
            connection.commit()
        finally:
            connection.close()
    except Exception:                                          # noqa: BLE001
        return


def observe(reading, *, plan_id: str = "", reopened: bool = False,
            run: Optional[Mapping[str, Any]] = None) -> None:
    """Every event one page view implies, from what actually happened.

    Derived from the reading and the run rather than from the route's belief
    about them: a handler that recorded `SAVED` before the write could report a
    save that failed.
    """
    run = run or {}

    if reopened:
        record(PLAN_REOPENED, plan_id=plan_id)
    else:
        record(PLAN_COMPILED, plan_id=plan_id,
               executable=bool(reading.executable),
               open_question_count=len(reading.questions))

    for refusal in reading.refusals:
        record(PLAN_REFUSED, plan_id=plan_id,
               refusal_code=CAPABILITY_REFUSED,
               capability=getattr(refusal, "dimension", ""),
               named_reason=getattr(refusal, "kind", ""))

    # Exactly one of these per execution. A run that emitted both would make
    # "results" and "refusals" sum to more than the executions that happened,
    # and every ratio drawn from them would be wrong in a way nothing shows.
    if run:
        if run.get("result") is not None:
            record(PLAN_RESULT_SHOWN, plan_id=plan_id)
        else:
            record(PLAN_REFUSED, plan_id=plan_id,
                   refusal_code=_execution_refusal_code(run))


def observe_save(reading, plan_id: str) -> None:
    """One event, with the open count as a number.

    The first version emitted a second event carrying the list of open field
    names. A count is what a cohort can be compared on; the names are already
    on the persisted artifact, where they belong to one plan rather than to a
    statistic.
    """
    record(PLAN_SAVED, plan_id=plan_id,
           executable=bool(reading.executable),
           settled_field_count=len(reading.settled),
           open_question_count=len(reading.questions))


def every_event() -> Sequence[Mapping[str, Any]]:
    connection = _connect()
    try:
        rows = connection.execute(
            "SELECT event_id, at, kind, plan_id, detail FROM pilot_events "
            "ORDER BY at").fetchall()
    finally:
        connection.close()
    return [{"event_id": r[0], "at": r[1], "kind": r[2], "plan_id": r[3],
             **json.loads(r[4])} for r in rows]


def summary() -> Mapping[str, Any]:
    """What the pilot can honestly report.

    Counts and names. Every number here is a thing that happened, and none of
    them is a claim about what anyone thought of it.
    """
    from collections import Counter

    events = every_event()
    refusals = Counter(e.get("capability", "") for e in events
                       if e["kind"] == PLAN_REFUSED and e.get("capability"))
    return {
        "events": len(events),
        "by_kind": dict(Counter(e["kind"] for e in events)),
        "by_profile": dict(Counter(
            f"{e.get('parser_mode')}/{e.get('pilot_reader')}" for e in events)),
        "refused_capabilities": dict(refusals.most_common()),
        "plans_compiled": sum(1 for e in events if e["kind"] == PLAN_COMPILED),
        "plans_saved": sum(1 for e in events if e["kind"] == PLAN_SAVED),
        "plans_saved_with_questions_open": sum(
            1 for e in events if e["kind"] == PLAN_SAVED
            and e.get("open_question_count")),
        "plans_reopened": len({e["plan_id"] for e in events
                               if e["kind"] == PLAN_REOPENED}),
        "results": sum(1 for e in events if e["kind"] == PLAN_RESULT_SHOWN),
        "refusals_by_code": dict(Counter(
            e.get("refusal_code", "") for e in events
            if e["kind"] == PLAN_REFUSED)),
        "not_measured_here": [
            "whether a refusal was understood",
            "whether the evidence made the result more trustworthy",
            "where someone gave up and why",
            "what they expected before they started",
        ],
    }
