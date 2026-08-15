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

#: Kept in step with `db/schema.py` and the migration that creates this table.
#:
#: This statement is why the table existed in production and in no migration:
#: it ran on first use, so the table appeared partway through a deployment's
#: life and the schema-parity preflight refused at the next restart. The table
#: is now declared and migrated; this remains so a fresh database is usable
#: before migrations run, and `test_no_undeclared_tables` is what keeps the two
#: from drifting apart.
SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_plans (
    plan_id       TEXT NOT NULL,
    owner         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    text          TEXT NOT NULL,
    artifact      TEXT NOT NULL,
    PRIMARY KEY (owner, plan_id)
)
"""

def PILOT_OWNER() -> str:
    """Whose workspace this request is in.

    A function rather than a constant, and that is the whole change: it was
    the literal `"pilot"` for every request, so every account that registered
    saw every other account's plans. The column existed and always held one
    value.

    Named as it was so the call sites read the same; `owner.current` is the
    thing that decides.
    """
    from .owner import current

    return current()


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


def _deployment_record() -> dict:
    """Which branch produced this plan, as a recorded fact.

    `compiled_by=quantify-mission@1` already implies the runtime path, and
    `single_witness` already implies the profile — but both are *inferences* a
    later analyst has to make, and an inference is not the same as a statement.
    Recording the declared mode makes the branch selection itself evidence, so
    a pilot conclusion can say which execution path produced a plan rather than
    reconstructing it from what the plan happens to contain.

    Read through the resolved context, like everything else here.
    """
    from ..deploy.context import current

    model = current().model
    return {"parser_mode": model.mode.value,
            "pilot_reader": model.pilot_reader.value,
            "model": model.model or ""}


def save(reading: PilotReading) -> str:
    from datetime import datetime, timezone

    plan_id = plan_id_for(reading)
    artifact = reading.to_json()
    artifact["text"] = reading.text
    artifact["deployment"] = _deployment_record()

    connection = _connect()
    try:
        connection.execute(
            "DELETE FROM pilot_plans WHERE plan_id = ? AND owner = ?",
            (plan_id, PILOT_OWNER()))
        connection.execute(
            "INSERT INTO pilot_plans "
            "(plan_id, owner, created_at, text, artifact) "
            "VALUES (?, ?, ?, ?, ?)",
            (plan_id, PILOT_OWNER(), datetime.now(timezone.utc).isoformat(),
             reading.text, json.dumps(artifact)))
        connection.commit()
    finally:
        connection.close()
    return plan_id


def load(plan_id: str) -> Optional[Mapping[str, Any]]:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT artifact FROM pilot_plans "
            "WHERE plan_id = ? AND owner = ?",
            (plan_id, PILOT_OWNER())).fetchone()
    finally:
        connection.close()
    return None if row is None else json.loads(row["artifact"])


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
    return [{"plan_id": r["plan_id"], "created_at": r["created_at"],
             **json.loads(r["artifact"])} for r in rows]
