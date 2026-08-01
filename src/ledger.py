"""Typed ledgers — the runtime state behind every published number.

Four tables, all append-only:

* **methodology_version** — what was published, when, and derived from what.
* **run** — one backtest execution, pinned to a methodology version and a manifest.
* **performance** — a figure, its class, its disclosure, and its trial ordinal.
* **erratum** — a correction that supersedes without deleting.

Two invariants are enforced in the schema rather than in the UI, because a
convention that lives only in a template gets bypassed by the first new consumer:

1. **No performance record without a performance class**, and the disclosure is
   attached to the row. A number cannot be served away from its caveat.
2. **Trial ordinals are assigned by the ledger**, not reported by the caller. The
   platform is the only party that can count honestly, because it is the only
   party that sees the variants that were tried and discarded.

SQLite is used deliberately for Release 1: it is a file, it needs no
infrastructure, and the schema is portable to Postgres when multi-tenancy
arrives. Every column type here exists in both.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .methodology.spec import REQUIRED_DISCLOSURE, Methodology, PerformanceClass

DEFAULT_DB = Path("data/quantify.db")

#: What changed. Kept closed — an open vocabulary would let the taxonomy drift
#: back into free text.
CORRECTION_TYPES = frozenset({"NUMERICAL", "METHODOLOGICAL", "INTERPRETIVE"})

#: Why it changed.
CAUSE_TYPES = frozenset({"DATA", "EXECUTION", "EVALUATION", "STATISTICAL", "PUBLICATION"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS methodology_version (
    version_id      TEXT PRIMARY KEY,
    concept_id      TEXT NOT NULL,
    concept         TEXT NOT NULL,
    version         INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    title           TEXT NOT NULL,
    derived_from    TEXT,
    change_rationale TEXT NOT NULL,
    risk_classification TEXT NOT NULL,
    deprecation_date TEXT,
    payload         TEXT NOT NULL,
    published_at    TEXT NOT NULL,
    UNIQUE (concept, version)
);

CREATE TABLE IF NOT EXISTS evaluation_protocol (
    protocol_id     TEXT PRIMARY KEY,
    concept_id      TEXT NOT NULL,
    name            TEXT NOT NULL,
    version         INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    title           TEXT NOT NULL,
    snapshot_hash   TEXT,
    holdout_sealed  INTEGER NOT NULL,
    derived_from    TEXT,
    change_rationale TEXT NOT NULL,
    payload         TEXT NOT NULL,
    published_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS holdout_unlock (
    unlock_id       TEXT PRIMARY KEY,
    protocol_id     TEXT NOT NULL,
    reason          TEXT NOT NULL,
    authorized_by   TEXT NOT NULL,
    unlocked_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS compatibility_result (
    compatibility_id TEXT PRIMARY KEY,
    concept         TEXT NOT NULL,
    version_id      TEXT NOT NULL,
    protocol_id     TEXT NOT NULL,
    trial_id        TEXT NOT NULL,
    compatible      INTEGER NOT NULL,
    blockers        TEXT NOT NULL,
    assessed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    run_id          TEXT PRIMARY KEY,
    version_id      TEXT NOT NULL REFERENCES methodology_version(version_id),
    protocol_id     TEXT NOT NULL,
    protocol_hash   TEXT NOT NULL,
    trial_id        TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    trial_ordinal   INTEGER NOT NULL,
    manifest        TEXT NOT NULL,
    manifest_digest TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    notes           TEXT,
    -- Persisted so a run is an artifact rather than a rendering. Without these
    -- a "run page" could only re-derive what the methodology page already shows,
    -- and Discovery would have to re-execute rather than traverse.
    result_status   TEXT,
    diagnostics     TEXT,
    execution_audit TEXT,
    assessment      TEXT,
    policy_evaluation TEXT,
    publication_decision TEXT,
    evidence_emitted TEXT
);

CREATE TABLE IF NOT EXISTS performance (
    performance_id  TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES run(run_id),
    version_id      TEXT NOT NULL REFERENCES methodology_version(version_id),
    protocol_id     TEXT NOT NULL,
    protocol_hash   TEXT NOT NULL,
    performance_class TEXT NOT NULL,
    disclosure      TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL NOT NULL,
    period_start    TEXT,
    period_end      TEXT,
    cost_model      TEXT NOT NULL,
    trials_at_publication INTEGER NOT NULL,
    superseded_by   TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS erratum (
    erratum_id      TEXT PRIMARY KEY,
    version_id      TEXT,
    title           TEXT NOT NULL,
    -- Three questions, three columns. One label cannot answer all of them:
    --   correction_type  what changed
    --   cause_type       why it changed
    --   severity         how consequential it was
    -- No UNKNOWN member: a system that insists on declared meaning should not
    -- introduce an undeclared one for its own convenience.
    correction_type TEXT NOT NULL,
    cause_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    summary         TEXT NOT NULL,
    supersedes      TEXT NOT NULL,
    document_path   TEXT,
    published_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_version ON run(version_id);
CREATE INDEX IF NOT EXISTS idx_perf_version ON performance(version_id);
CREATE INDEX IF NOT EXISTS idx_mv_concept ON methodology_version(concept);
"""


def _dump(payload) -> Optional[str]:
    """Serialize a persisted run component, or None when absent."""
    return json.dumps(payload, default=str) if payload is not None else None


def _now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


@dataclass(frozen=True)
class PerformanceRecord:
    performance_id: str
    run_id: str
    version_id: str
    protocol_id: str
    protocol_hash: str
    performance_class: PerformanceClass
    disclosure: str
    metric: str
    value: float
    cost_model: str
    trials_at_publication: int
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    superseded_by: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "performance_id": self.performance_id,
            "run_id": self.run_id,
            "version_id": self.version_id,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "performance_class": self.performance_class.value,
            "disclosure": self.disclosure,
            "metric": self.metric,
            "value": self.value,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "cost_model": self.cost_model,
            "trials_at_publication": self.trials_at_publication,
            "superseded_by": self.superseded_by,
        }


class Ledger:
    """Append-only store for methodologies, runs, performance and errata."""

    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- methodology ------------------------------------------------------

    def publish_methodology(self, m: Methodology) -> str:
        """Record a version. Republishing the same content is a no-op.

        Republishing *different* content under an existing version id is an
        error: a version id names an immutable artifact, and results already cite
        it.
        """
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM methodology_version WHERE version_id = ?",
                (m.version_id,),
            ).fetchone()
            if existing:
                if existing["content_hash"] == m.content_hash:
                    return m.version_id
                raise ValueError(
                    f"{m.version_id} already published with a different content hash. "
                    "Published versions are immutable — mint a new version instead."
                )
            conn.execute(
                """INSERT INTO methodology_version
                   (version_id, concept_id, concept, version, content_hash, title,
                    derived_from, change_rationale, risk_classification,
                    deprecation_date, payload, published_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m.version_id, m.concept_id, m.concept, m.version, m.content_hash,
                    m.title, m.derived_from, m.change_rationale, m.risk_classification,
                    m.deprecation_date, json.dumps(m.to_json()), _now(),
                ),
            )
        return m.version_id

    def get_methodology(self, version_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM methodology_version WHERE version_id = ?", (version_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_methodologies(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version_id, concept, version, title, content_hash, "
                "derived_from, deprecation_date, published_at "
                "FROM methodology_version ORDER BY concept, version"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- runs and the trial counter --------------------------------------

    def trial_count(self, concept: str, *, dsr_countable_only: bool = False) -> int:
        """Distinct trial identities attempted across this lineage.

        The unit is a **materially distinct analytical configuration**, not a
        methodology version and not a version × protocol product. Repeating one
        configuration is a reproducibility check, not a new search; varying an
        embargo is a new search even though the methodology is unchanged.

        `dsr_countable_only` applies the stated policy in
        :data:`trial.DSR_COUNTABLE_OUTCOMES` — configurations blocked before
        execution revealed nothing about the data and are excluded from the DSR
        denominator, while still being recorded as attempted.
        """
        from .trial import DSR_COUNTABLE_OUTCOMES

        query = """SELECT DISTINCT run.trial_id, run.outcome FROM run
                   JOIN methodology_version USING (version_id)
                   WHERE methodology_version.concept = ?"""
        with self._conn() as conn:
            rows = conn.execute(query, (concept,)).fetchall()

        ids = {
            r["trial_id"]
            for r in rows
            if not dsr_countable_only or r["outcome"] in {o.value for o in DSR_COUNTABLE_OUTCOMES}
        }

        if not dsr_countable_only:
            with self._conn() as conn:
                blocked = conn.execute(
                    "SELECT DISTINCT trial_id FROM compatibility_result "
                    "WHERE concept = ? AND compatible = 0",
                    (concept,),
                ).fetchall()
            ids |= {r["trial_id"] for r in blocked}

        return len(ids)

    def trial_breakdown(self, concept: str) -> Dict[str, Any]:
        """Where the search went.

        Reports attempted configurations and the DSR-countable subset separately,
        because they answer different questions: how hard was this searched, and
        how much of that search could have inflated the maximum Sharpe.
        """
        with self._conn() as conn:
            runs = [
                dict(r)
                for r in conn.execute(
                    """SELECT run.trial_id, run.version_id, run.protocol_id, run.outcome
                       FROM run JOIN methodology_version USING (version_id)
                       WHERE methodology_version.concept = ?""",
                    (concept,),
                ).fetchall()
            ]
            blocked = [
                dict(r)
                for r in conn.execute(
                    """SELECT trial_id, version_id, protocol_id FROM compatibility_result
                       WHERE concept = ? AND compatible = 0""",
                    (concept,),
                ).fetchall()
            ]

        outcomes: Dict[str, int] = {}
        for r in runs:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

        attempted = {r["trial_id"] for r in runs} | {b["trial_id"] for b in blocked}
        return {
            "attempted_trials": len(attempted),
            "dsr_countable_trials": self.trial_count(concept, dsr_countable_only=True),
            "blocked_before_execution": len({b["trial_id"] for b in blocked}),
            "executions": len(runs),
            "repeat_executions": len(runs) - len({r["trial_id"] for r in runs}),
            "outcomes": outcomes,
            "distinct_methodology_versions": len(
                {r["version_id"] for r in runs} | {b["version_id"] for b in blocked}
            ),
            "distinct_protocols": len(
                {r["protocol_id"] for r in runs} | {b["protocol_id"] for b in blocked}
            ),
            "policy": (
                "A trial is a materially distinct configuration: "
                "(methodology, protocol, objective, data partition, execution "
                "assumptions). Pairings blocked before execution are recorded as "
                "attempted but excluded from the DSR denominator — they revealed "
                "nothing about the data."
            ),
        }

    def record_compatibility(
        self,
        *,
        compatibility_id: str,
        concept: str,
        version_id: str,
        protocol_id: str,
        trial_id: str,
        compatible: bool,
        blockers: List[Dict[str, Any]],
    ) -> str:
        """Persist a pairing assessment, compatible or not."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO compatibility_result
                   (compatibility_id, concept, version_id, protocol_id, trial_id,
                    compatible, blockers, assessed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    compatibility_id, concept, version_id, protocol_id, trial_id,
                    int(compatible), json.dumps(blockers), _now(),
                ),
            )
        return compatibility_id

    def list_compatibility(self, concept: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM compatibility_result"
        params: tuple = ()
        if concept:
            query += " WHERE concept = ?"
            params = (concept,)
        query += " ORDER BY assessed_at DESC"
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        for r in rows:
            r["blockers"] = json.loads(r["blockers"])
            r["compatible"] = bool(r["compatible"])
        return rows

    # ---- evaluation protocols --------------------------------------------

    def publish_protocol(self, protocol) -> str:
        """Record an evaluation protocol. Same immutability rule as methodologies."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM evaluation_protocol WHERE protocol_id = ?",
                (protocol.protocol_id,),
            ).fetchone()
            if existing:
                if existing["content_hash"] == protocol.content_hash:
                    return protocol.protocol_id
                raise ValueError(
                    f"{protocol.protocol_id} already published with a different content "
                    "hash. Published protocols are immutable — results cite them."
                )
            conn.execute(
                """INSERT INTO evaluation_protocol
                   (protocol_id, concept_id, name, version, content_hash, title,
                    snapshot_hash, holdout_sealed, derived_from, change_rationale,
                    payload, published_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    protocol.protocol_id, protocol.concept_id, protocol.name,
                    protocol.version, protocol.content_hash, protocol.title,
                    protocol.data_snapshot.content_hash,
                    int(protocol.holdout.sealed), protocol.derived_from,
                    protocol.change_rationale, json.dumps(protocol.to_json()), _now(),
                ),
            )
        return protocol.protocol_id

    def list_protocols(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT protocol_id, name, version, title, content_hash, "
                "snapshot_hash, holdout_sealed, published_at "
                "FROM evaluation_protocol ORDER BY name, version"
            ).fetchall()
        return [dict(r) for r in rows]

    def record_holdout_unlock(
        self, *, unlock_id: str, protocol_id: str, reason: str, authorized_by: str
    ) -> str:
        """Log a holdout unlock. Once per protocol, and refused thereafter."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT unlock_id FROM holdout_unlock WHERE protocol_id = ?",
                (protocol_id,),
            ).fetchone()
            if existing:
                raise ValueError(
                    f"{protocol_id} holdout was already unlocked by "
                    f"{existing['unlock_id']!r}. A sealed holdout opens once."
                )
            conn.execute(
                """INSERT INTO holdout_unlock
                   (unlock_id, protocol_id, reason, authorized_by, unlocked_at)
                   VALUES (?,?,?,?,?)""",
                (unlock_id, protocol_id, reason, authorized_by, _now()),
            )
        return unlock_id

    def list_holdout_unlocks(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM holdout_unlock ORDER BY unlocked_at DESC"
                ).fetchall()
            ]

    def record_run(
        self,
        *,
        run_id: str,
        version_id: str,
        protocol_id: str,
        protocol_hash: str,
        manifest: Dict[str, Any],
        manifest_digest: str,
        trial_id: Optional[str] = None,
        outcome: str = "completed",
        status: str = "completed",
        notes: str = "",
        result_status: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        execution_audit: Optional[Dict[str, Any]] = None,
        assessment: Optional[Dict[str, Any]] = None,
        policy_evaluation: Optional[Dict[str, Any]] = None,
        publication_decision: Optional[Dict[str, Any]] = None,
        evidence_emitted: Optional[List[str]] = None,
    ) -> int:
        """Record a run and return its trial ordinal.

        Requires both artifacts: a run that does not name the protocol it used
        cannot be reproduced, because the protocol carries the costs, the lag, the
        grid and the data snapshot.

        The ordinal is assigned here and is never a caller input — that is the
        whole mechanism. A researcher can under-report how many variants they
        tried; a platform that owns the execution path cannot.
        """
        meta = self.get_methodology(version_id)
        if meta is None:
            raise ValueError(f"unknown methodology version {version_id!r}")
        # Fall back to a configuration-derived id so a caller that omits it still
        # gets identity semantics rather than one-trial-per-run counting.
        trial_id = trial_id or (
            "trial_"
            + hashlib.sha256(f"{version_id}|{protocol_hash}".encode()).hexdigest()[:32]
        )
        ordinal = self.trial_count(meta["concept"]) + 1
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO run
                   (run_id, version_id, protocol_id, protocol_hash, trial_id, outcome,
                    trial_ordinal, manifest, manifest_digest, started_at, status, notes,
                    result_status, diagnostics, execution_audit, assessment,
                    policy_evaluation, publication_decision, evidence_emitted)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, version_id, protocol_id, protocol_hash, trial_id, outcome,
                    ordinal, json.dumps(manifest), manifest_digest, _now(), status, notes,
                    _dump(result_status), _dump(diagnostics), _dump(execution_audit),
                    _dump(assessment), _dump(policy_evaluation),
                    _dump(publication_decision), _dump(evidence_emitted),
                ),
            )
        return ordinal

    def list_runs(self, version_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT run_id, version_id, protocol_id, protocol_hash, trial_ordinal, "
            "manifest_digest, started_at, status FROM run"
        )
        params: tuple = ()
        if version_id:
            query += " WHERE version_id = ?"
            params = (version_id,)
        query += " ORDER BY started_at DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # ---- performance ------------------------------------------------------

    def record_performance(
        self,
        *,
        performance_id: str,
        run_id: str,
        version_id: str,
        protocol_id: str,
        protocol_hash: str,
        performance_class: PerformanceClass,
        metric: str,
        value: float,
        cost_model: str,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> PerformanceRecord:
        """Record a figure with its class, disclosure, trial count and both hashes.

        A performance row names the methodology *and* the protocol, because
        `methodology + protocol = performance`. Citing only the methodology would
        make the figure irreproducible — the costs, lag, grid and data snapshot
        all live in the protocol.

        The disclosure is looked up from the class and stored on the row, so it
        travels with the number into every API response and export.
        """
        if not isinstance(performance_class, PerformanceClass):
            raise TypeError(
                "performance_class must be a PerformanceClass — an unclassified "
                "number cannot be rendered safely"
            )
        meta = self.get_methodology(version_id)
        if meta is None:
            raise ValueError(f"unknown methodology version {version_id!r}")

        record = PerformanceRecord(
            performance_id=performance_id,
            run_id=run_id,
            version_id=version_id,
            protocol_id=protocol_id,
            protocol_hash=protocol_hash,
            performance_class=performance_class,
            disclosure=REQUIRED_DISCLOSURE[performance_class],
            metric=metric,
            value=value,
            cost_model=cost_model,
            trials_at_publication=self.trial_count(meta["concept"]),
            period_start=period_start,
            period_end=period_end,
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO performance
                   (performance_id, run_id, version_id, protocol_id, protocol_hash,
                    performance_class, disclosure, metric, value, period_start,
                    period_end, cost_model, trials_at_publication, superseded_by,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.performance_id, record.run_id, record.version_id,
                    record.protocol_id, record.protocol_hash,
                    record.performance_class.value, record.disclosure, record.metric,
                    record.value, record.period_start, record.period_end,
                    record.cost_model, record.trials_at_publication, None, _now(),
                ),
            )
        return record

    def list_performance(
        self, version_id: Optional[str] = None, include_superseded: bool = False
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM performance"
        clauses: List[str] = []
        params: List[Any] = []
        if version_id:
            clauses.append("version_id = ?")
            params.append(version_id)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # ---- errata -----------------------------------------------------------

    def publish_erratum(
        self,
        *,
        erratum_id: str,
        title: str,
        correction_type: str,
        cause_type: str,
        severity: str,
        summary: str,
        supersedes: List[str],
        version_id: Optional[str] = None,
        document_path: Optional[str] = None,
    ) -> str:
        """Publish a correction and mark the superseded records.

        Superseded performance is *flagged*, never deleted — the erratum links to
        what it replaced, which is the behaviour the whole product claims.

        `correction_type` and `cause_type` are separate and both required. Both
        current errata are NUMERICAL/DATA, but a future INTERPRETIVE/STATISTICAL
        case — a conclusion that was wrong because a statistic was misread, with
        every figure correct — needs a different response, and one label cannot
        express it.
        """
        if correction_type not in CORRECTION_TYPES:
            raise ValueError(
                f"correction_type must be one of {sorted(CORRECTION_TYPES)}, "
                f"got {correction_type!r}"
            )
        if cause_type not in CAUSE_TYPES:
            raise ValueError(
                f"cause_type must be one of {sorted(CAUSE_TYPES)}, got {cause_type!r}"
            )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO erratum
                   (erratum_id, version_id, title, correction_type, cause_type,
                    severity, summary, supersedes, document_path, published_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    erratum_id, version_id, title, correction_type, cause_type,
                    severity, summary, json.dumps(supersedes), document_path, _now(),
                ),
            )
            for perf_id in supersedes:
                conn.execute(
                    "UPDATE performance SET superseded_by = ? WHERE performance_id = ?",
                    (erratum_id, perf_id),
                )
        return erratum_id

    def list_errata(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM erratum ORDER BY published_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["supersedes"] = json.loads(d["supersedes"])
            out.append(d)
        return out

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """One run, with everything persisted about it.

        A run is an artifact: its diagnostics, audit, assessment, policy verdict
        and publication decision are stored at execution time, not recomputed on
        read. Recomputing would mean a run page shows today's code applied to
        yesterday's execution — which is a rendering, not a record.
        """
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM run WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None

        out = dict(row)
        for column in (
            "manifest", "result_status", "diagnostics", "execution_audit",
            "assessment", "policy_evaluation", "publication_decision",
            "evidence_emitted",
        ):
            if out.get(column):
                out[column] = json.loads(out[column])
        return out

    def runs_emitting_evidence(self, evidence_ref: str) -> List[Dict[str, Any]]:
        """Which runs produced a given piece of evidence.

        The link that lets Discovery traverse from a claim, through the evidence
        bearing on it, back to the execution that generated it — without
        re-running anything.
        """
        return [
            r for r in (self.get_run(x["run_id"]) for x in self.list_runs())
            if r and evidence_ref in (r.get("evidence_emitted") or [])
        ]
