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
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..mission.boundary import scan_for_personal_data
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
    status               TEXT NOT NULL
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
    result_runs     TEXT
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


class NotSaveable(ValueError):
    """A plan with unconfirmed choices cannot be saved.

    Saving turns a placeholder into a commitment the user never made, which is
    the same principle that stops an unrealized declaration from publishing.
    """


class WorkspaceStore:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._add_missing_columns(conn)
            self._relax_not_null(conn)
            self._widen_primary_keys(conn)

    @staticmethod
    def _widen_primary_keys(conn: sqlite3.Connection) -> None:
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
    def _relax_not_null(conn: sqlite3.Connection) -> None:
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
    def _add_missing_columns(conn: sqlite3.Connection) -> None:
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
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One connection across several writes, committed or rolled back once.

        The apply path persists runs and then a worksheet revision that cites
        them. Committing those separately leaves a window where an accepted edit
        has produced runs and no revision — an orphaned run that looks like
        history and belongs to nothing.
        """
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
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
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # Inside a transaction every write joins it rather than committing on
        # its own, so a failure halfway through rolls the whole edit back.
        joined = getattr(self, "_tx", None)
        if joined is not None:
            yield joined
            return
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
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
            conn.execute(
                """INSERT OR REPLACE INTO plan
                   (plan_id, owner, title, scenario, intent, stated_text,
                    saved_at, rule_hash, content_hash, parse)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (plan_id, owner, scenario.name, json.dumps(payload), intent_id,
                 stated_text, saved_at, scenario.rule_hash, scenario.content_hash,
                 json.dumps(parse) if parse is not None else None),
            )
        return plan_id

    def record_run(self, *, run_id: str, plan_id: str, ran_at: str,
                   result: Dict[str, Any], comparison: Dict[str, Any]) -> str:
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
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO plan_run
                   (run_id, plan_id, ran_at, result, comparison) VALUES (?,?,?,?,?)""",
                (run_id, plan_id, ran_at, json.dumps(result), json.dumps(comparison)),
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
                 json.dumps(payload), worksheet.canonical_hash,
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
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

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
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def worksheet_revisions(self, worksheet_id: str,
                            owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM worksheet WHERE worksheet_id = ? AND owner = ?"
                " ORDER BY revision", (worksheet_id, owner)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

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
                                proposal_id: Optional[str] = None) -> int:
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
                    planner_version, chain_hash, created_at, proposal_id, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (intent.intent_id, worksheet_id, owner, intent.source_revision,
                 sequence,
                 intent.instruction if store_instruction else None,
                 instruction_hash,
                 json.dumps(intent.to_json()),
                 intent.edit_effect.value, intent.selection_basis.value,
                 intent.repetition_signature.key(),
                 json.dumps(list(intent.related_prior_intents)),
                 int(intent.results_visible), intent.alternatives_generated,
                 intent.trial_effect, planner_version, chain_hash, created_at,
                 proposal_id, "PLANNED"))
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
                 "structured_request": json.loads(row["structured_request"]),
                 "related_prior": json.loads(row["related_prior"] or "[]")}
                for row in rows]

    def link_intent_proposal(self, intent_id: str, owner: str, *,
                             proposal_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE worksheet_intent SET proposal_id = ?, status = ? "
                "WHERE intent_id = ? AND owner = ?",
                (proposal_id, "PROPOSED", intent_id, owner))

    def save_worksheet_proposal(self, *, proposal_id: str, owner: str,
                                worksheet_id: str, proposal,
                                created_at: str) -> str:
        """Record a worksheet proposal as PROPOSED. Immutable from here."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO worksheet_proposal
                   (proposal_id, owner, worksheet_id, source_revision, status,
                    payload, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (proposal_id, owner, worksheet_id, proposal.source_revision,
                 "PROPOSED", json.dumps(proposal.to_json()), created_at))
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
        return {**dict(row), "payload": json.loads(row["payload"]),
                "result_runs": json.loads(row["result_runs"] or "[]")}

    def resolve_worksheet_proposal(self, proposal_id: str, owner: str, *,
                                   status: str, resolved_at: str,
                                   actor: str = "",
                                   result_revision: Optional[int] = None,
                                   result_runs: Sequence[str] = ()) -> None:
        """Record the outcome. The reviewed diff is never rewritten."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE worksheet_proposal
                   SET status = ?, resolved_at = ?, actor = ?,
                       result_revision = ?, result_runs = ?
                   WHERE proposal_id = ? AND owner = ? AND status = 'PROPOSED'""",
                (status, resolved_at, actor, result_revision,
                 json.dumps(list(result_runs)), proposal_id, owner))

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

    def get_run(self, run_id: str, owner: str) -> Optional[Dict[str, Any]]:
        """One run by id, scoped by owner through its plan.

        Exists because a worksheet pins an exact run. Resolving through
        `runs_for(...)[0]` would hand back the *newest* run for the plan, which
        is how a saved worksheet silently starts showing figures it never cited.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT plan_run.* FROM plan_run
                   JOIN plan ON plan.plan_id = plan_run.plan_id
                   WHERE plan_run.run_id = ? AND plan.owner = ?""",
                (run_id, owner)).fetchone()
        if row is None:
            return None
        return {**dict(row), "result": json.loads(row["result"]),
                "comparison": json.loads(row["comparison"])}

    def runs_for(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        if self.get_plan(plan_id, owner) is None:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM plan_run WHERE plan_id = ? ORDER BY ran_at DESC",
                (plan_id,),
            ).fetchall()
        return [
            {**dict(r), "result": json.loads(r["result"]),
             "comparison": json.loads(r["comparison"])}
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
                 json.dumps(proposal.to_json()), proposal.generated_at,
                 proposal.status.value),
            )
        return proposal.proposal_id

    def list_proposals(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM proposal WHERE plan_id = ? AND owner = ?
                   ORDER BY generated_at DESC""", (plan_id, owner)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # ---- observations -----------------------------------------------------

    def save_observation(self, *, owner: str, observation) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO observation
                   (observation_id, plan_id, owner, observed_at, payload)
                   VALUES (?,?,?,?,?)""",
                (observation.artifact_id, observation.plan_id, owner,
                 observation.observed_at, json.dumps(observation.to_json())),
            )
        return observation.artifact_id

    def list_observations(self, plan_id: str, owner: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM observation WHERE plan_id = ? AND owner = ?
                   ORDER BY observed_at DESC""", (plan_id, owner)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> Dict[str, Any]:
        record = dict(row)
        record["scenario"] = json.loads(record["scenario"])
        # Absent for plans saved before stage 1 could involve a model. Those
        # recompile deterministically, which is what they always did.
        stored = record.get("parse")
        record["parse"] = json.loads(stored) if stored else None
        return record
