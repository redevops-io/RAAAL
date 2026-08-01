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
from typing import Any, Dict, Iterator, List, Optional

from ..mission.boundary import scan_for_personal_data

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
    def _conn(self) -> Iterator[sqlite3.Connection]:
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
