"""Where a pilot plan lives, and what a stored plan is.

Deliberately its own table rather than a column on the existing plans. The
legacy store holds `stated_text` and recompiles from it on every read, which is
the behaviour the runtime replaces — a plan whose meaning is re-derived from
prose each time it is opened. Sharing the table would have meant one row that
means two different things depending on which interpreter wrote it, and the
first person to read a plan a year from now could not tell which.

**What is stored is the artifact, not a rendering of it.** The pinned intent,
whole; the settled record with its provenance; the refusals by name. Reopening
recompiles from that and never re-reads the sentence, which is why the sentence
is stored *beside* the intent rather than as the thing the plan is made of.
"""
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Mapping, Optional

from ..db.engine import Database
from .pilot import PilotReading

SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_plans (
    plan_id       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    text          TEXT NOT NULL,
    artifact      TEXT NOT NULL
)
"""


def _database() -> Database:
    """Through `db.engine`, not a driver.

    The first version imported `sqlite3` directly and
    `test_error_surface.test_every_driver_reference_is_declared` caught it.
    Declaring the piercing with a reason would have satisfied the letter of
    that check and missed its point: there was no reason. `Database` already
    resolves the target, picks the dialect, and hands back a connection whose
    failures the translation layer understands — piercing it here would have
    meant this table's errors reached a user untranslated while every other
    table's did not.
    """
    from ..deploy.context import current

    return Database(current().database.url)


def _connect():
    database = _database()
    connection = database.connect()
    connection.execute(SCHEMA)
    return connection


def plan_id_for(reading: PilotReading) -> str:
    """Derived from the intent, so the same confirmed plan is the same plan.

    Not a random id. Two submissions that produced the identical pinned intent
    *are* the same plan, and giving them different names would make the replay
    property unobservable — the whole point being that the artifact, not the
    request, is what identifies it.
    """
    if reading.intent is None:
        return "draft-" + sha256(reading.text.encode()).hexdigest()[:12]
    return "plan-" + sha256(reading.intent.intent_hash.encode()).hexdigest()[:12]


def save(reading: PilotReading) -> str:
    from datetime import datetime, timezone

    plan_id = plan_id_for(reading)
    artifact = reading.to_json()
    artifact["text"] = reading.text

    connection = _connect()
    try:
        connection.execute(
            "DELETE FROM pilot_plans WHERE plan_id = ?", (plan_id,))
        connection.execute(
            "INSERT INTO pilot_plans (plan_id, created_at, text, artifact) "
            "VALUES (?, ?, ?, ?)",
            (plan_id, datetime.now(timezone.utc).isoformat(),
             reading.text, json.dumps(artifact)))
        connection.commit()
    finally:
        connection.close()
    return plan_id


def load(plan_id: str) -> Optional[Mapping[str, Any]]:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT artifact FROM pilot_plans WHERE plan_id = ?",
            (plan_id,)).fetchone()
    finally:
        connection.close()
    return None if row is None else json.loads(row[0])


def every_plan() -> list:
    """For the pilot's own instrumentation — what people submitted and what the
    runtime made of it. Read-only, and not exposed on a route."""
    connection = _connect()
    try:
        rows = connection.execute(
            "SELECT plan_id, created_at, artifact FROM pilot_plans "
            "ORDER BY created_at").fetchall()
    finally:
        connection.close()
    return [{"plan_id": r[0], "created_at": r[1], **json.loads(r[2])}
            for r in rows]
