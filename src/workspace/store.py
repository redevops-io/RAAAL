"""Private storage for the scenario workspace.

A separate file, a separate schema, and a separate module from the public
ledger. The boundary decision said separate data stores, and a shared database
with a `visibility` column is not that: one forgotten predicate in one query is
the whole difference, and it will be written by someone who does not know the
rule exists.

The store additionally refuses to write anything that fails the boundary check,
so a plan cannot be persisted carrying a reference that would leak if it were
ever exported.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ..db.engine import Database, Dialect
from ..db.decimals import (
    DecimalDrift,
    Money,
    same_quantity,
    same_value,
    to_decimal,
)
from ..db.types import Json, loads
from ..mission.boundary import scan_for_personal_data
from ..runtime.base import canonical_hash
from .intent_chain import chain_link

DEFAULT_PATH = Path("data/workspace.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plan (
    plan_id      TEXT PRIMARY KEY,
    owner        TEXT NOT NULL,
    title        TEXT NOT NULL,
    scenario     TEXT NOT NULL,
    intent       TEXT,
    stated_text  TEXT NOT NULL,
    saved_at     TEXT NOT NULL,
    rule_hash    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    -- The stage 1 parse this plan was compiled from. Pinned rather than
    -- re-derived: stage 1 may involve a language model, and recompiling a saved
    -- plan against a model that has since changed would silently alter a plan
    -- the user already read and confirmed.
    parse        TEXT
);
CREATE TABLE IF NOT EXISTS proposal (
    proposal_id TEXT PRIMARY KEY,
    plan_id     TEXT NOT NULL,
    owner       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    status      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observation (
    observation_id TEXT PRIMARY KEY,
    plan_id     TEXT NOT NULL,
    owner       TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload     TEXT NOT NULL
);
-- What was asked, before anything changed. Durable on its own, not folded into
-- the proposal it produced.
--
-- The planner's protections are history-dependent: "add 63-day volatility" is
-- analytical the first time and parameter tuning the fourth, and the repetition
-- signature is what stops repeated tuning hiding behind rephrasing. Held only in
-- a request, that history dies between calls — a user could try three windows
-- across three requests and each would arrive looking like the first. Trial
-- accounting that resets is worse than none, because it reports a small number
-- rather than no number.
--
-- `instruction` is nullable on purpose. The durable semantic record is the
-- structured intent and `instruction_hash`; the raw sentence may carry holdings,
-- salary or employer detail and is subject to a stricter retention policy than
-- the classification derived from it.
CREATE TABLE IF NOT EXISTS worksheet_intent (
    intent_id            TEXT PRIMARY KEY,
    worksheet_id         TEXT NOT NULL,
    owner                TEXT NOT NULL,
    source_revision      INTEGER NOT NULL,
    sequence             INTEGER NOT NULL,
    instruction          TEXT,
    instruction_hash     TEXT NOT NULL,
    structured_request   TEXT NOT NULL,
    edit_effect          TEXT NOT NULL,
    selection_basis      TEXT NOT NULL,
    repetition_signature TEXT NOT NULL,
    related_prior        TEXT NOT NULL,
    results_visible      INTEGER NOT NULL,
    alternatives         INTEGER NOT NULL,
    -- Nullable deliberately. NULL means the planner could not read the
    -- instruction, which is not the same as it costing nothing: an
    -- unclassified request may have asked for one chart or for forty.
    trial_effect         INTEGER,
    planner_version      TEXT NOT NULL,
    -- Each row chains to its predecessor. Editing a prior intent's
    -- classification, or deleting one from the middle, breaks every successor's
    -- hash — so a trial total derived from a doctored chain is detectably
    -- derived from a doctored chain rather than quietly smaller.
    chain_hash           TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    proposal_id          TEXT,
    status               TEXT NOT NULL,
    -- A reference into operational telemetry, which expires on its own
    -- schedule. Nullable, and nothing may require it to resolve: a trace that
    -- has aged out must not make a stored intent unreadable.
    trace_id             TEXT
);
-- Ordering within a worksheet must be unique and gapless. Two intents claiming
-- one position make the chain ambiguous, and an ambiguous chain cannot support
-- a trial total anyone should rely on.
CREATE UNIQUE INDEX IF NOT EXISTS worksheet_intent_sequence
    ON worksheet_intent (worksheet_id, owner, sequence);
-- One row per worksheet *revision*. Revisions are never edited, so the primary
-- key spans the id and the revision: an UPDATE that lost a revision would erase
-- the history that revisions exist to keep.
-- Worksheet proposals are immutable. Acceptance creates new artifacts and
-- records the outcome here; it never rewrites the diff that was reviewed.
--
-- Named apart from `proposal`, which is the mission forward-tracking artifact.
-- `CREATE TABLE IF NOT EXISTS` on a name already taken is a silent no-op, so
-- the columns would simply not have existed.
CREATE TABLE IF NOT EXISTS worksheet_proposal (
    proposal_id     TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,
    worksheet_id    TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    status          TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    actor           TEXT,
    result_revision INTEGER,
    result_runs     TEXT,
    trace_id        TEXT
);
CREATE TABLE IF NOT EXISTS worksheet (
    worksheet_id  TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    owner         TEXT NOT NULL,
    payload       TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    -- Owner is part of the identity, not a filter applied afterwards.
    --
    -- Keyed on (worksheet_id, revision) alone, a second owner could not create
    -- a worksheet whose id another tenant already held: the write was refused,
    -- and the refusal answered a question the requester was not entitled to
    -- ask. Reads were correctly scoped, so nothing leaked on the way out — the
    -- oracle was on the way in.
    PRIMARY KEY (owner, worksheet_id, revision)
);
-- Confirmation-screen telemetry. Structure now, conclusions later: intent
-- cannot be inferred without users, but the first sessions are the ones worth
-- measuring and they only happen once.
-- Forward tracking, as three independent records. Inputs and conclusions are
-- stored separately so the conclusion can be re-derived and compared, the same
-- two-layer check the result context uses for presentability.
--
-- Owner is in every key from the start. These rows carry employer names, grant
-- references and compensation quantities, so a cross-tenant existence leak here
-- is more sensitive than the worksheet-id one that prompted the rule.
CREATE TABLE IF NOT EXISTS planned_event (
    owner            TEXT NOT NULL,
    worksheet_id     TEXT NOT NULL,
    planned_event_id TEXT NOT NULL,
    plan_revision    INTEGER NOT NULL,
    grant_ref        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    expected_effective_date TEXT NOT NULL,
    asset            TEXT,
    expected_quantity REAL,
    expected_value   REAL,
    payload          TEXT NOT NULL,
    matching_policy_version TEXT NOT NULL,
    source_ref       TEXT,
    content_hash     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (owner, worksheet_id, planned_event_id)
);
-- `effective_date` and `observed_at` stay apart all the way through storage. A
-- vest reported in July may have settled in June, and collapsing them would
-- make an on-time vest look late for as long as the record survives.
CREATE TABLE IF NOT EXISTS observed_event (
    owner             TEXT NOT NULL,
    worksheet_id      TEXT NOT NULL,
    observed_event_id TEXT NOT NULL,
    kind              TEXT NOT NULL,
    effective_date    TEXT NOT NULL,
    observed_at       TEXT NOT NULL,
    asset             TEXT,
    quantity          REAL,
    value             REAL,
    payload           TEXT NOT NULL,
    evidence_refs     TEXT NOT NULL DEFAULT '[]',
    source            TEXT NOT NULL,
    supersedes        TEXT,
    content_hash      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (owner, worksheet_id, observed_event_id)
);
CREATE TABLE IF NOT EXISTS event_reconciliation (
    owner             TEXT NOT NULL,
    worksheet_id      TEXT NOT NULL,
    reconciliation_id TEXT NOT NULL,
    planned_event_id  TEXT,
    -- Nullable: pending, overdue and confirmed-missing rows have no
    -- observation, and a placeholder would read as one.
    observed_event_id TEXT,
    status            TEXT NOT NULL,
    payload           TEXT NOT NULL,
    matching_policy_version TEXT NOT NULL,
    superseded_by     TEXT,
    content_hash      TEXT NOT NULL,
    derived_at        TEXT NOT NULL,
    PRIMARY KEY (owner, worksheet_id, reconciliation_id)
);
CREATE INDEX IF NOT EXISTS reconciliation_worksheet
    ON event_reconciliation (owner, worksheet_id, derived_at);
CREATE TABLE IF NOT EXISTS confirmation_event (
    event_id   TEXT PRIMARY KEY,
    owner      TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    path       TEXT,
    field      TEXT,
    provenance TEXT,
    original_value TEXT,
    final_value TEXT,
    reason     TEXT,
    compiler_version TEXT,
    defaults_ref TEXT
);
CREATE TABLE IF NOT EXISTS plan_run (
    run_id     TEXT PRIMARY KEY,
    plan_id    TEXT NOT NULL,
    ran_at     TEXT NOT NULL,
    result     TEXT NOT NULL,
    comparison TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES plan (plan_id)
);
"""


#: Columns added after the first release. Applied on open, in order.
_ADDED_COLUMNS = (
    ("plan", "parse", "parse TEXT"),
    ("worksheet_intent", "trace_id", "trace_id TEXT"),
    ("worksheet_proposal", "trace_id", "trace_id TEXT"),
)

#: Constraints relaxed after the table shipped. SQLite cannot ALTER a NOT NULL
#: away, so the table is rebuilt. Listed as (table, column) pairs that must be
#: nullable; a database created before the change would otherwise raise on the
#: first unclassified instruction — a failure that cannot reproduce on a fresh
#: checkout, which is the worst kind.
_RELAXED_NOT_NULL = (
    ("worksheet_intent", "trial_effect"),
)

#: Tables whose primary key gained a column after shipping. SQLite cannot ALTER
#: a primary key, so the table is rebuilt. `worksheet` was keyed on
#: (worksheet_id, revision) with no owner, which made a write refusal reveal
#: that another tenant held that id.
_WIDENED_PRIMARY_KEY = (
    ("worksheet", ("owner", "worksheet_id", "revision")),
)


#: The denormalized decimal columns and the payload field each one mirrors.
#: Read by `verify_decimal_columns`, which re-derives the comparison from the
#: stored payload rather than trusting that the write did it.
MIRRORED_DECIMALS = {
    "planned_event": {"expected_quantity": "expected_gross_shares",
                      "expected_value": "expected_value"},
    "observed_event": {"quantity": "gross_shares", "value": "value"},
}


def _mirror(payload: Mapping[str, Any], field: str, value: Any,
            table: str) -> Money:
    """Bind a decimal column, having checked it against the payload it copies.

    These columns exist so a value can be filtered, ordered and aggregated in
    the database. That makes them a second answer to a question the hashed
    payload already answers, and a second answer that can disagree is worse
    than not having one — the disagreement would surface in a query result or a
    threshold comparison, which is the least visible place for it.

    The payload stays authoritative. This does not reconcile the two; it refuses
    to write a row where they differ.
    """
    if not same_value(payload.get(field), value):
        raise DecimalDrift(
            f"{table}.{field}: the value being stored ({value!r}) and the "
            f"payload field it mirrors ({payload.get(field)!r}) are different "
            "quantities. The payload is authoritative and the column is a copy "
            "of it; writing both would give the database two answers.")
    return Money(value)


def _with_decimals(record: Dict[str, Any], table: str,
                   **extra: Any) -> Dict[str, Any]:
    """Normalize a row's decimal columns, whichever dialect produced it.

    PostgreSQL returns a `Decimal` from NUMERIC and SQLite returns canonical
    text. Callers get a `Decimal` from both, so nothing downstream has to know
    which engine it read from — the property the differing column types exist
    to preserve.
    """
    for column in MIRRORED_DECIMALS[table]:
        record[column] = to_decimal(record.get(column))
    record.update(extra)
    return record


#: Tables holding an immutable artifact body with a hash over it.
HASHED_ARTIFACTS = {
    "planned_event": "planned_event_id",
    "observed_event": "observed_event_id",
    "event_reconciliation": "reconciliation_id",
}


def verify_content_hashes(store, table: str,
                          owner: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows whose stored payload no longer matches the hash taken over it.

    The narrowest of the three checks, and deliberately so. It answers only
    "is this body the one that was written", and says nothing about whether a
    mirrored column agrees with it or whether a derived conclusion still
    follows from it. Each of those has its own verifier, because one broad
    check that happened to catch all three would hide the absence of the other
    two the day it stopped.

    Recomputes from the stored payload rather than trusting anything alongside
    it: a tampered payload with an untouched hash is the case this exists for.
    """
    identifier = HASHED_ARTIFACTS[table]
    sql = f"SELECT * FROM {table}"
    params: Sequence[Any] = ()
    if owner is not None:
        sql += " WHERE owner = ?"
        params = (owner,)

    corrupted: List[Dict[str, Any]] = []
    with store._conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    for row in rows:
        payload = loads(row["payload"], {})
        recomputed = canonical_hash(payload)
        if recomputed != row["content_hash"]:
            corrupted.append({"table": table, "id": row[identifier],
                              "owner": row["owner"],
                              "stored_hash": row["content_hash"],
                              "recomputed_hash": recomputed})
    return corrupted


def verify_decimal_columns(store, table: str,
                           owner: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows whose decimal column disagrees with the payload it mirrors.

    Re-derives the comparison from what is stored rather than trusting the
    write that put it there. A check that shared code with the writer would
    agree with it by construction — the same reason `verify_deleted` reads the
    schema instead of the deletion it is verifying.

    Returns the drifted rows rather than raising, because an integrity sweep
    wants all of them and not just the first.
    """
    drifted: List[Dict[str, Any]] = []
    sql = f"SELECT * FROM {table}"
    params: Sequence[Any] = ()
    if owner is not None:
        sql += " WHERE owner = ?"
        params = (owner,)
    with store._conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    for row in rows:
        payload = loads(row["payload"], {})
        for column, field in MIRRORED_DECIMALS[table].items():
            stored, mirrored = row[column], payload.get(field)
            try:
                # Numeric comparison, not spelling: a NUMERIC column pads to
                # its declared scale, so the stored value legitimately reads
                # back with trailing zeros the payload does not carry.
                agrees = same_quantity(stored, mirrored)
            except Exception:
                agrees = False
            if not agrees:
                drifted.append({"table": table, "column": column,
                                "field": field, "stored": stored,
                                "payload": mirrored,
                                "row": {k: row[k] for k in
                                        ("owner", "worksheet_id")}})
    return drifted


class NotSaveable(ValueError):
    """A plan with unconfirmed choices cannot be saved.

    Saving turns a placeholder into a commitment the user never made, which is
    the same principle that stops an unrealized declaration from publishing.
    """


class WorkspaceStore:
    """The workspace, on whichever database this instance was pointed at.

    Accepts a filesystem path (SQLite — tests, local development, the standalone
    demo), a database URL, or nothing at all, in which case
    `QUANTIFY_DATABASE_URL` decides and a local SQLite file is the fallback.
    A deployed pilot is required to be PostgreSQL by `src.db.guard`, not by this
    constructor: a store that refused SQLite outright could not be unit-tested.
    """

    def __init__(self, target: Path | str | None = None) -> None:
        self.db = Database(target if target is not None else DEFAULT_PATH)
        self.path = self.db.path
        self.db.create_all()
        if self.db.dialect is Dialect.SQLITE:
            # Legacy repair, SQLite only. PostgreSQL gets its schema from
            # Alembic, where an ordered revision does this properly and leaves a
            # record that it happened.
            with self._conn() as conn:
                self._add_missing_columns(conn)
                self._relax_not_null(conn)
                self._widen_primary_keys(conn)

    @staticmethod
    def _widen_primary_keys(conn) -> None:
        """Rebuild any table whose primary key gained a column."""
        for table, expected in _WIDENED_PRIMARY_KEY:
            columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
            present = tuple(c["name"] for c in sorted(
                (c for c in columns if c["pk"]), key=lambda c: c["pk"]))
            if present == tuple(expected):
                continue

            names = ", ".join(c["name"] for c in columns)
            ddl = next(iter(_SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table} (")[1]
                            .split(");")))
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}__old")
            conn.execute(f"CREATE TABLE {table} ({ddl})")
            conn.execute(f"INSERT INTO {table} ({names}) "
                         f"SELECT {names} FROM {table}__old")
            conn.execute(f"DROP TABLE {table}__old")

    @staticmethod
    def _relax_not_null(conn) -> None:
        """Rebuild any table whose column must now accept NULL."""
        for table, column in _RELAXED_NOT_NULL:
            columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if not any(c["name"] == column and c["notnull"] for c in columns):
                continue

            names = ", ".join(c["name"] for c in columns)
            ddl = next(iter(_SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table} (")[1]
                            .split(");")))
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}__old")
            conn.execute(f"CREATE TABLE {table} ({ddl})")
            conn.execute(f"INSERT INTO {table} ({names}) "
                         f"SELECT {names} FROM {table}__old")
            conn.execute(f"DROP TABLE {table}__old")

    @staticmethod
    def _add_missing_columns(conn) -> None:
        """`CREATE TABLE IF NOT EXISTS` does nothing to a table that exists.

        An existing workspace database would keep its old shape and every insert
        naming a new column would fail at runtime — a deployment failure that
        cannot reproduce on a fresh checkout, which is the worst kind.
        """
        for table, column, ddl in _ADDED_COLUMNS:
            present = {row["name"] for row in
                       conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """One connection across several writes, committed or rolled back once.

        The apply path persists runs and then a worksheet revision that cites
        them. Committing those separately leaves a window where an accepted edit
        has produced runs and no revision — an orphaned run that looks like
        history and belongs to nothing.
        """
        conn = self.db.connect()
        previous, self._tx = getattr(self, "_tx", None), conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._tx = previous
            conn.close()

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        # Inside a transaction every write joins it rather than committing on
        # its own, so a failure halfway through rolls the whole edit back.
        joined = getattr(self, "_tx", None)
        if joined is not None:
            yield joined
            return
        conn = self.db.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_plan(self, *, plan_id: str, owner: str, scenario, stated_text: str,
                  saved_at: str, intent_id: Optional[str] = None,
                  parse: Optional[Dict[str, Any]] = None) -> str:
        if not scenario.is_runnable:
            raise NotSaveable(
                f"{scenario.artifact_id} contradicts itself: "
                + "; ".join(scenario.self_conflicts())
            )
        if not scenario.provenance.is_complete:
            raise NotSaveable(
                f"{scenario.artifact_id} still has unconfirmed inferences or open "
                "questions. It can be simulated to show its shape; saving it "
                "would commit the user to choices they have not made"
            )

        payload = scenario.to_json()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM plan WHERE plan_id = ? AND owner = ?",
                (plan_id, owner)).fetchone()
            if existing is not None:
                if existing["content_hash"] == scenario.content_hash:
                    return plan_id           # idempotent redelivery
                raise NotSaveable(
                    f"plan {plan_id} is already stored with different contents. "
                    "The compiled scenario and the stage 1 parse are pinned — "
                    "replacing them would alter a plan the user has already "
                    "read and confirmed. Save it under a new id instead")
            conn.execute(
                """INSERT INTO plan
                   (plan_id, owner, title, scenario, intent, stated_text,
                    saved_at, rule_hash, content_hash, parse)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (plan_id, owner, scenario.name, Json(payload), intent_id,
                 stated_text, saved_at, scenario.rule_hash, scenario.content_hash,
                 Json(parse) if parse is not None else None),
            )
        return plan_id

    def record_run(self, *, run_id: str, plan_id: str, ran_at: str,
                   result: Dict[str, Any], comparison: Dict[str, Any],
                   owner: Optional[str] = None) -> str:
        """Persist a historical run so its verdict survives later changes.

        Same reason the public ledger stores verdicts rather than recomputing
        them: a plan revisited next year must show the result it actually got.

        A result with no modelling scope is refused. The scope is what says which
        costs and taxes the figure excludes, and a stored number that has lost it
        will eventually be read as though it excluded nothing.
        """
        if not result.get("modelling_scope"):
            raise NotSaveable(
                f"run {run_id} carries no modelling scope. A recorded figure "
                "without a statement of what it excludes will be read as "
                "excluding nothing"
            )

        # A run that declares it used RSU mechanics must carry the context that
        # says whether its figure is complete. Storing it without one leaves a
        # number whose caveats exist only in a function that has returned.
        if result.get("requires_rsu_context") and not result.get("rsu_context"):
            raise NotSaveable(
                f"run {run_id} declares RSU mechanics and carries no result "
                "context. The diagnostics that decide whether this figure is "
                "presentable would exist nowhere after this write")

        if result.get("rsu_context"):
            from ..mission.rsu_result import validate as _validate_context

            _validate_context(result["rsu_context"])
        with self._conn() as conn:
            if owner is None:
                # Derived from the plan rather than defaulted. A run belongs to
                # whoever owns the plan it ran, and there is exactly one such
                # owner now that `plan` is keyed by both — so this resolves or
                # it refuses, and never guesses.
                rows = conn.execute(
                    "SELECT owner FROM plan WHERE plan_id = ?",
                    (plan_id,)).fetchall()
                if len(rows) != 1:
                    raise NotSaveable(
                        f"run {run_id} names plan {plan_id!r}, which resolves "
                        f"to {len(rows)} owners. A run must belong to exactly "
                        "one; pass `owner` explicitly")
                owner = rows[0]["owner"]
            digest = canonical_hash({"result": result, "comparison": comparison})
            existing = conn.execute(
                "SELECT result, comparison FROM plan_run "
                "WHERE run_id = ? AND owner = ?", (run_id, owner)).fetchone()
            if existing is not None:
                stored = canonical_hash({
                    "result": loads(existing["result"], {}),
                    "comparison": loads(existing["comparison"], {})})
                if stored == digest:
                    return run_id            # idempotent redelivery
                raise NotSaveable(
                    f"run {run_id} is already stored with a different result. "
                    "A run records the verdict a plan actually got; replacing "
                    "it would make a saved worksheet show a figure it never "
                    "cited. Record a new run instead")
            conn.execute(
                """INSERT INTO plan_run
                   (owner, run_id, plan_id, ran_at, result, comparison)
                   VALUES (?,?,?,?,?,?)""",
                (owner, run_id, plan_id, ran_at, Json(result), Json(comparison)),
            )
        return run_id

    # ---- worksheets ------------------------------------------------------

    def save_worksheet(self, worksheet) -> str:
        """Write one revision. Never updates an existing one.

        `INSERT OR REPLACE` would let a second write at the same revision
        silently overwrite the first, which is exactly the history a revision
        exists to preserve.
        """
        payload = worksheet.to_json()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT canonical_hash FROM worksheet "
                "WHERE owner = ? AND worksheet_id = ? AND revision = ?",
                (worksheet.owner_id, worksheet.worksheet_id,
                 worksheet.revision)).fetchone()
            if existing is not None:
                if existing["canonical_hash"] == worksheet.canonical_hash:
                    return worksheet.worksheet_id      # idempotent redelivery
                raise NotSaveable(
                    f"{worksheet.worksheet_id} revision {worksheet.revision} is "
                    "already stored with different contents. Revisions are "
                    "immutable; make a new one rather than moving this")
            conn.execute(
                """INSERT INTO worksheet (worksheet_id, revision, owner, payload,
                                          canonical_hash, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (worksheet.worksheet_id, worksheet.revision, worksheet.owner_id,
                 Json(payload), worksheet.canonical_hash,
                 worksheet.created_at or ""))
        return worksheet.worksheet_id

    def worksheet_for_scenario(self, scenario_ref: str,
                               owner: str) -> Optional[Dict[str, Any]]:
        """This owner's latest worksheet for one scenario, if any.

        Scoped by owner in the query. Two tenants may hold worksheets for
        identically-named scenarios and neither may observe the other's.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM worksheet WHERE owner = ? "
                "AND json_extract(payload, '$.scenario_ref') = ? "
                "ORDER BY revision DESC LIMIT 1",
                (owner, scenario_ref)).fetchone()
        return {**dict(row), "payload": loads(row["payload"])} if row else None

    def get_worksheet(self, worksheet_id: str, owner: str,
                      revision: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """The named revision, or the latest. Scoped by owner in the query."""
        with self._conn() as conn:
            if revision is None:
                row = conn.execute(
                    "SELECT * FROM worksheet WHERE worksheet_id = ? AND owner = ?"
                    " ORDER BY revision DESC LIMIT 1",
                    (worksheet_id, owner)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM worksheet WHERE worksheet_id = ? AND owner = ?"
                    " AND revision = ?", (worksheet_id, owner, revision)).fetchone()
        return {**dict(row), "payload": loads(row["payload"])} if row else None

    def worksheet_revisions(self, worksheet_id: str,
                            owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM worksheet WHERE worksheet_id = ? AND owner = ?"
                " ORDER BY revision", (worksheet_id, owner)).fetchall()
        return [{**dict(r), "payload": loads(r["payload"])} for r in rows]

    def record_confirmation_event(self, *, event_id: str, owner: str,
                                  occurred_at: str, kind: str, **fields) -> str:
        """Structure now, conclusions later.

        Deliberately records *what changed*, never why. Intent cannot be
        inferred from a value edit — "you misunderstood me", "I changed my mind"
        and "I had not said" are different product signals and look identical
        here. `reason` is filled only when a user is explicitly asked.
        """
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO confirmation_event
                   (event_id, owner, occurred_at, kind, path, field, provenance,
                    original_value, final_value, reason, compiler_version,
                    defaults_ref)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (event_id, owner, occurred_at, kind, fields.get("path"),
                 fields.get("field"), fields.get("provenance"),
                 fields.get("original_value"), fields.get("final_value"),
                 fields.get("reason"), fields.get("compiler_version"),
                 fields.get("defaults_ref")))
        return event_id

    def confirmation_events(self, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM confirmation_event WHERE owner = ?"
                " ORDER BY occurred_at", (owner,)).fetchall()
        return [dict(r) for r in rows]

    # ---- proposals -------------------------------------------------------

    # ---- worksheet intents ------------------------------------------------

    def append_worksheet_intent(self, *, worksheet_id: str, owner: str,
                                intent, created_at: str,
                                planner_version: str,
                                instruction_hash: str,
                                store_instruction: bool = False,
                                proposal_id: Optional[str] = None,
                                trace_id: Optional[str] = None) -> int:
        """Add one intent to a worksheet's chain and return its position.

        The position is derived here, inside the write, rather than supplied by
        the caller. A caller-chosen sequence is a caller-chosen history, and the
        planner's protections are exactly the thing a caller has an incentive to
        renumber.

        `store_instruction` is off by default. The classification is the durable
        record; the sentence it came from is personal data with a shorter life.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS last FROM worksheet_intent "
                "WHERE worksheet_id = ? AND owner = ?",
                (worksheet_id, owner)).fetchone()
            sequence = int(row["last"]) + 1
            previous = conn.execute(
                "SELECT chain_hash FROM worksheet_intent "
                "WHERE worksheet_id = ? AND owner = ? ORDER BY sequence DESC "
                "LIMIT 1", (worksheet_id, owner)).fetchone()
            chain_hash = chain_link(
                previous["chain_hash"] if previous else "", intent)
            conn.execute(
                """INSERT INTO worksheet_intent
                   (intent_id, worksheet_id, owner, source_revision, sequence,
                    instruction, instruction_hash, structured_request,
                    edit_effect, selection_basis, repetition_signature,
                    related_prior, results_visible, alternatives, trial_effect,
                    planner_version, chain_hash, created_at, proposal_id,
                    status, trace_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (intent.intent_id, worksheet_id, owner, intent.source_revision,
                 sequence,
                 intent.instruction if store_instruction else None,
                 instruction_hash,
                 Json(intent.to_json()),
                 intent.edit_effect.value, intent.selection_basis.value,
                 intent.repetition_signature.key(),
                 Json(list(intent.related_prior_intents)),
                 int(intent.results_visible), intent.alternatives_generated,
                 intent.trial_effect, planner_version, chain_hash, created_at,
                 proposal_id, "PLANNED", trace_id))
        return sequence

    def worksheet_intents(self, worksheet_id: str, owner: str, *,
                          before_sequence: Optional[int] = None
                          ) -> List[Dict[str, Any]]:
        """The chain for one worksheet, in order, scoped to its owner.

        Owner is part of the query rather than checked afterwards. A history
        filtered after loading is a history that was loaded.
        """
        clause = "WHERE worksheet_id = ? AND owner = ?"
        params: List[Any] = [worksheet_id, owner]
        if before_sequence is not None:
            clause += " AND sequence < ?"
            params.append(before_sequence)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM worksheet_intent {clause} ORDER BY sequence",
                params).fetchall()
        return [{**dict(row),
                 "structured_request": loads(row["structured_request"]),
                 "related_prior": loads(row["related_prior"], [])}
                for row in rows]

    def link_intent_proposal(self, intent_id: str, owner: str, *,
                             proposal_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE worksheet_intent SET proposal_id = ?, status = ? "
                "WHERE intent_id = ? AND owner = ?",
                (proposal_id, "PROPOSED", intent_id, owner))

    # ---- forward tracking -------------------------------------------------

    def record_planned_event(self, *, owner: str, worksheet_id: str,
                             event, plan_revision: int, created_at: str,
                             matching_policy_version: str) -> str:
        """Store one expectation, tied to the plan revision that produced it.

        Immutable. An identical write is redelivery; the same id with a
        different body is a conflict, because a prediction that changed after
        the fact is not a prediction.
        """
        payload = event.to_json()
        digest = canonical_hash(payload)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM planned_event WHERE owner = ? "
                "AND worksheet_id = ? AND planned_event_id = ?",
                (owner, worksheet_id, event.event_id)).fetchone()
            if existing is not None:
                if existing["content_hash"] == digest:
                    return event.event_id
                raise NotSaveable(
                    f"planned event {event.event_id} already exists with a "
                    "different body. An expectation that changed after the "
                    "fact is not an expectation; record a new plan revision")
            conn.execute(
                """INSERT INTO planned_event
                   (owner, worksheet_id, planned_event_id, plan_revision,
                    grant_ref, kind, expected_effective_date, asset,
                    expected_quantity, expected_value, payload,
                    matching_policy_version, source_ref, content_hash,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (owner, worksheet_id, event.event_id, plan_revision,
                 event.grant_ref, event.kind, event.expected_date,
                 event.employer_asset,
                 _mirror(payload, "expected_gross_shares",
                         event.expected_gross_shares, "planned_event"),
                 _mirror(payload, "expected_value", event.expected_value,
                         "planned_event"),
                 Json(payload),
                 matching_policy_version, event.source_declaration, digest,
                 created_at))
        return event.event_id

    def record_observed_event(self, *, owner: str, worksheet_id: str,
                              event, created_at: str,
                              supersedes: Optional[str] = None) -> str:
        """Store one report. A correction is a new row, never an overwrite.

        Overwriting the first would erase the fact that a correction happened,
        which is part of the audit trail rather than noise in it.
        """
        payload = event.to_json()
        digest = canonical_hash(payload)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM observed_event WHERE owner = ? "
                "AND worksheet_id = ? AND observed_event_id = ?",
                (owner, worksheet_id, event.observation_id)).fetchone()
            if existing is not None:
                if existing["content_hash"] == digest:
                    return event.observation_id
                raise NotSaveable(
                    f"observation {event.observation_id} already exists with a "
                    "different body. Record a correcting observation that "
                    "supersedes it rather than rewriting what was reported")
            conn.execute(
                """INSERT INTO observed_event
                   (owner, worksheet_id, observed_event_id, kind,
                    effective_date, observed_at, asset, quantity, value,
                    payload, evidence_refs, source, supersedes, content_hash,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (owner, worksheet_id, event.observation_id, event.kind,
                 event.effective_date, event.observed_date,
                 event.employer_asset,
                 _mirror(payload, "gross_shares", event.gross_shares,
                         "observed_event"),
                 _mirror(payload, "value", event.value, "observed_event"),
                 Json(payload),
                 Json([event.evidence_ref] if event.evidence_ref else []),
                 event.source, supersedes, digest, created_at))
        return event.observation_id

    def record_reconciliation(self, *, owner: str, worksheet_id: str,
                              reconciliation) -> str:
        payload = reconciliation.to_json()
        digest = canonical_hash(payload)
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO event_reconciliation
                   (owner, worksheet_id, reconciliation_id, planned_event_id,
                    observed_event_id, status, payload,
                    matching_policy_version, content_hash, derived_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (owner, worksheet_id, reconciliation.reconciliation_id,
                 reconciliation.planned_ref, reconciliation.observed_ref,
                 reconciliation.status.value, Json(payload),
                 reconciliation.matching_policy_version, digest,
                 reconciliation.derived_at))
        return reconciliation.reconciliation_id

    def planned_events(self, worksheet_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM planned_event WHERE owner = ? AND "
                "worksheet_id = ? ORDER BY expected_effective_date",
                (owner, worksheet_id)).fetchall()
        return [_with_decimals(dict(r), "planned_event",
                               payload=loads(r["payload"])) for r in rows]

    def observed_events(self, worksheet_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM observed_event WHERE owner = ? AND "
                "worksheet_id = ? ORDER BY effective_date, created_at",
                (owner, worksheet_id)).fetchall()
        return [_with_decimals(dict(r), "observed_event",
                               payload=loads(r["payload"]),
                               evidence_refs=loads(r["evidence_refs"], []))
                for r in rows]

    def reconciliations(self, worksheet_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM event_reconciliation WHERE owner = ? AND "
                "worksheet_id = ? ORDER BY derived_at",
                (owner, worksheet_id)).fetchall()
        return [{**dict(r), "payload": loads(r["payload"])} for r in rows]

    def save_worksheet_proposal(self, *, proposal_id: str, owner: str,
                                worksheet_id: str, proposal,
                                created_at: str,
                                trace_id: Optional[str] = None) -> str:
        """Record a worksheet proposal as PROPOSED. Immutable from here."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO worksheet_proposal
                   (proposal_id, owner, worksheet_id, source_revision, status,
                    payload, created_at, trace_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (proposal_id, owner, worksheet_id, proposal.source_revision,
                 "PROPOSED", Json(proposal.to_json()), created_at,
                 trace_id))
        return proposal_id

    def get_worksheet_proposal(self, proposal_id: str,
                               owner: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM worksheet_proposal "
                "WHERE proposal_id = ? AND owner = ?",
                (proposal_id, owner)).fetchone()
        if row is None:
            return None
        return {**dict(row), "payload": loads(row["payload"]),
                "result_runs": loads(row["result_runs"], [])}

    def lock_worksheet_proposal(self, proposal_id: str,
                                owner: str) -> Optional[Dict[str, Any]]:
        """Take the proposal row and hold it for the rest of the transaction.

        Must be called inside `transaction()`. Every check that authorizes an
        acceptance has to happen after this and re-read what it authorizes:
        a check performed before the lock describes state another session is
        still free to change, and two sessions passing the same pre-lock check
        is exactly how one review produced two acceptances.

        SQLite has no `FOR UPDATE` and does not need one — it admits a single
        writer, so the transaction is already exclusive. The portable authority
        is the conditional update in `resolve_worksheet_proposal`; this is
        PostgreSQL reinforcing it by making the loser wait rather than race.
        """
        clause = (" FOR UPDATE" if self.db.dialect is Dialect.POSTGRESQL
                  else "")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM worksheet_proposal "
                "WHERE proposal_id = ? AND owner = ?" + clause,
                (proposal_id, owner)).fetchone()
        if row is None:
            return None
        return {**dict(row), "payload": loads(row["payload"]),
                "result_runs": loads(row["result_runs"], [])}

    def resolve_worksheet_proposal(self, proposal_id: str, owner: str, *,
                                   status: str, resolved_at: str,
                                   actor: str = "",
                                   result_revision: Optional[int] = None,
                                   result_runs: Sequence[str] = ()) -> int:
        """Record the outcome, and report whether it was this call that did.

        The `status = 'PROPOSED'` predicate makes this a conditional state
        transition: at most one caller can move a proposal out of PROPOSED,
        whatever else is happening concurrently. Returning the row count is
        what makes that useful — the predicate was already here, and the
        result was thrown away, so a losing caller updated nothing and
        continued as though it had won.

        The reviewed diff is never rewritten.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE worksheet_proposal
                   SET status = ?, resolved_at = ?, actor = ?,
                       result_revision = ?, result_runs = ?
                   WHERE proposal_id = ? AND owner = ? AND status = 'PROPOSED'""",
                (status, resolved_at, actor, result_revision,
                 Json(list(result_runs)), proposal_id, owner))
            return cursor.rowcount

    def list_plans(self, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM plan WHERE owner = ? ORDER BY saved_at DESC", (owner,)
            ).fetchall()
        return [self._hydrate(r) for r in rows]

    def get_plan(self, plan_id: str, owner: str) -> Optional[Dict[str, Any]]:
        """Scoped by owner at the query, not filtered afterwards.

        A get that fetches by id and checks ownership in Python is one early
        return away from serving someone else's plan.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM plan WHERE plan_id = ? AND owner = ?",
                (plan_id, owner),
            ).fetchone()
        return self._hydrate(row) if row else None

    @staticmethod
    def rsu_context_of(record: Mapping[str, Any]):
        """The stored context, validated, or a stated absence.

        Returns `None` for a run that never declared one. A caller must treat
        that as NOT_DECLARED rather than as clean — an older record's silence is
        evidence that nothing was recorded, not that nothing happened.
        """
        from ..mission.rsu_result import from_json as _context_from_json
        from ..mission.rsu_result import validate as _validate_context

        payload = (record.get("result") or {}).get("rsu_context")
        if not payload:
            return None
        _validate_context(payload)
        return _context_from_json(payload)

    def get_run(self, run_id: str, owner: str) -> Optional[Dict[str, Any]]:
        """One run by id, scoped by owner through its plan.

        Exists because a worksheet pins an exact run. Resolving through
        `runs_for(...)[0]` would hand back the *newest* run for the plan, which
        is how a saved worksheet silently starts showing figures it never cited.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM plan_run WHERE run_id = ? AND owner = ?",
                (run_id, owner)).fetchone()
        if row is None:
            return None
        return {**dict(row), "result": loads(row["result"]),
                "comparison": loads(row["comparison"])}

    def runs_for(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        # Owner is in the query, not a pre-check followed by an unscoped read.
        # The pre-check was correct and the read beneath it was not, which is
        # the shape that survives every refactor of the check.
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM plan_run WHERE plan_id = ? AND owner = ? "
                "ORDER BY ran_at DESC",
                (plan_id, owner),
            ).fetchall()
        return [
            {**dict(r), "result": loads(r["result"]),
             "comparison": loads(r["comparison"])}
            for r in rows
        ]

    # ---- proposals --------------------------------------------------------

    def save_proposal(self, *, owner: str, proposal) -> str:
        """Persist a proposal, including a resolved one.

        Resolutions are written as new rows keyed by the same id, so the current
        state is stored while the object that produced it stays immutable.
        Expired and ignored proposals are kept: neither is a failed record, and
        an expiry is the only evidence that a constraint cost something.
        """
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO proposal
                   (proposal_id, plan_id, owner, payload, generated_at, status)
                   VALUES (?,?,?,?,?,?)""",
                (proposal.proposal_id, proposal.plan_id, owner,
                 Json(proposal.to_json()), proposal.generated_at,
                 proposal.status.value),
            )
        return proposal.proposal_id

    def list_proposals(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM proposal WHERE plan_id = ? AND owner = ?
                   ORDER BY generated_at DESC""", (plan_id, owner)).fetchall()
        return [{**dict(r), "payload": loads(r["payload"])} for r in rows]

    # ---- observations -----------------------------------------------------

    def save_observation(self, *, owner: str, observation) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO observation
                   (observation_id, plan_id, owner, observed_at, payload)
                   VALUES (?,?,?,?,?)""",
                (observation.artifact_id, observation.plan_id, owner,
                 observation.observed_at, Json(observation.to_json())),
            )
        return observation.artifact_id

    def list_observations(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM observation WHERE plan_id = ? AND owner = ?
                   ORDER BY observed_at DESC""", (plan_id, owner)).fetchall()
        return [{**dict(r), "payload": loads(r["payload"])} for r in rows]

    @staticmethod
    def _hydrate(row: Mapping[str, Any]) -> Dict[str, Any]:
        record = dict(row)
        record["scenario"] = loads(record["scenario"])
        # Absent for plans saved before stage 1 could involve a model. Those
        # recompile deterministically, which is what they always did.
        record["parse"] = loads(record.get("parse"))
        return record
