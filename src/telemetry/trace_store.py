"""Storage for operational telemetry. A separate database, deliberately.

`workspace.db` holds immutable financial artifacts and is never pruned. This
holds spans and decisions and is pruned on a schedule. Sharing one file would
make retention a per-table convention that some future query forgets, and would
put deletable rows inside the transaction that writes permanent ones.

Nothing here is authoritative. If this file is missing, corrupt, or empty, every
financial path must behave identically — see `tests/test_telemetry.py`, which
deletes it mid-flight and asserts exactly that.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

DEFAULT_PATH = Path("data/trace.db")

#: How long operational telemetry survives. Financial artifacts have no
#: equivalent because they have no expiry.
DEFAULT_RETENTION_DAYS = 90

_SCHEMA = """
-- The correlation spine. A conversation outlives any one request, which is what
-- makes "show me every model interaction that eventually produced revision 17"
-- answerable — the artifact graph alone cannot reach back past the request that
-- wrote it.
CREATE TABLE IF NOT EXISTS trace (
    trace_id        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    tenant          TEXT NOT NULL,
    worksheet_id    TEXT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    status          TEXT NOT NULL,
    -- What this request eventually produced, by reference. The link from
    -- telemetry to artifacts points this way on purpose: an artifact that
    -- pointed at a trace it required would stop being replayable the day the
    -- trace expired.
    produced        TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS trace_conversation
    ON trace (tenant, conversation_id, started_at);
CREATE INDEX IF NOT EXISTS trace_worksheet
    ON trace (tenant, worksheet_id, started_at);

CREATE TABLE IF NOT EXISTS span (
    span_id      TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL,
    parent_id    TEXT,
    name         TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    duration_ms  REAL,
    status       TEXT NOT NULL,
    -- Structured fields and hashes only. No raw instruction text, no prompt or
    -- completion bodies: those may carry holdings, salary, employer or RSU
    -- detail, and a trace store is the wrong place for any of it.
    attributes   TEXT NOT NULL DEFAULT '{}',
    error        TEXT
);
CREATE INDEX IF NOT EXISTS span_trace ON span (trace_id, started_at);

CREATE TABLE IF NOT EXISTS decision (
    decision_id  TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL,
    kind         TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    reason       TEXT NOT NULL,
    evidence_refs TEXT NOT NULL DEFAULT '[]',
    confidence   REAL,
    alternatives TEXT NOT NULL DEFAULT '[]',
    at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_trace ON decision (trace_id);
CREATE INDEX IF NOT EXISTS decision_kind ON decision (kind, at);
"""


class TraceStore:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writing ---------------------------------------------------------

    def start_trace(self, *, trace_id: str, conversation_id: str,
                    request_id: str, tenant: str, started_at: str,
                    worksheet_id: Optional[str] = None) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trace
                   (trace_id, conversation_id, request_id, tenant, worksheet_id,
                    started_at, status)
                   VALUES (?,?,?,?,?,?,?)""",
                (trace_id, conversation_id, request_id, tenant, worksheet_id,
                 started_at, "OPEN"))
        return trace_id

    def end_trace(self, trace_id: str, *, ended_at: str, status: str,
                  produced: Sequence[str] = ()) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE trace SET ended_at = ?, status = ?, produced = ? "
                "WHERE trace_id = ?",
                (ended_at, status, json.dumps(list(produced)), trace_id))

    def record_span(self, span) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO span
                   (span_id, trace_id, parent_id, name, started_at, duration_ms,
                    status, attributes, error)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (span.span_id, span.trace_id, span.parent_id, span.name,
                 span.started_at, span.duration_ms, span.status,
                 json.dumps(dict(span.attributes)), span.error))
        return span.span_id

    def record_decision(self, decision) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO decision
                   (decision_id, trace_id, kind, outcome, reason, evidence_refs,
                    confidence, alternatives, at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (decision.decision_id, decision.trace_id, decision.kind.value,
                 decision.outcome, decision.reason,
                 json.dumps(list(decision.evidence_refs)), decision.confidence,
                 json.dumps(list(decision.alternatives_considered)),
                 decision.at))
        return decision.decision_id

    # ---- reading ---------------------------------------------------------

    def trace(self, trace_id: str, tenant: str) -> Optional[Dict[str, Any]]:
        """One trace with its spans and decisions. Scoped by tenant in SQL."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trace WHERE trace_id = ? AND tenant = ?",
                (trace_id, tenant)).fetchone()
            if row is None:
                return None
            spans = conn.execute(
                "SELECT * FROM span WHERE trace_id = ? ORDER BY started_at",
                (trace_id,)).fetchall()
            decisions = conn.execute(
                "SELECT * FROM decision WHERE trace_id = ? ORDER BY at",
                (trace_id,)).fetchall()
        return {
            **dict(row), "produced": json.loads(row["produced"] or "[]"),
            "spans": [{**dict(s), "attributes": json.loads(s["attributes"])}
                      for s in spans],
            "decisions": [{**dict(d),
                           "evidence_refs": json.loads(d["evidence_refs"]),
                           "alternatives": json.loads(d["alternatives"])}
                          for d in decisions],
        }

    def traces_for_conversation(self, conversation_id: str,
                                tenant: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trace WHERE conversation_id = ? AND tenant = ? "
                "ORDER BY started_at", (conversation_id, tenant)).fetchall()
        return [{**dict(r), "produced": json.loads(r["produced"] or "[]")}
                for r in rows]

    def traces_producing(self, artifact_ref: str,
                         tenant: str) -> List[Dict[str, Any]]:
        """Every trace that produced a given artifact.

        The question this store exists for: "show me every model interaction
        that eventually produced worksheet revision 17". Answerable only while
        the traces are still inside their retention window — which is why the
        artifact itself never depends on the answer.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trace WHERE tenant = ? "
                "AND EXISTS (SELECT 1 FROM json_each(trace.produced) "
                "            WHERE json_each.value = ?) "
                "ORDER BY started_at", (tenant, artifact_ref)).fetchall()
        return [{**dict(r), "produced": json.loads(r["produced"] or "[]")}
                for r in rows]

    def decisions_of_kind(self, kind: str, tenant: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT decision.* FROM decision JOIN trace USING (trace_id) "
                "WHERE decision.kind = ? AND trace.tenant = ? ORDER BY at",
                (kind, tenant)).fetchall()
        return [{**dict(r), "evidence_refs": json.loads(r["evidence_refs"]),
                 "alternatives": json.loads(r["alternatives"])} for r in rows]

    # ---- retention -------------------------------------------------------

    def purge_before(self, cutoff: str) -> Dict[str, int]:
        """Delete traces started before `cutoff`, with their spans and decisions.

        Telemetry expires; artifacts do not. Nothing here consults the workspace
        before deleting, because nothing in the workspace is allowed to need it.
        """
        with self._conn() as conn:
            stale = [r["trace_id"] for r in conn.execute(
                "SELECT trace_id FROM trace WHERE started_at < ?",
                (cutoff,)).fetchall()]
            if not stale:
                return {"traces": 0, "spans": 0, "decisions": 0}

            marks = ",".join("?" * len(stale))
            spans = conn.execute(
                f"SELECT COUNT(*) AS n FROM span WHERE trace_id IN ({marks})",
                stale).fetchone()["n"]
            decisions = conn.execute(
                f"SELECT COUNT(*) AS n FROM decision WHERE trace_id IN ({marks})",
                stale).fetchone()["n"]
            conn.execute(f"DELETE FROM span WHERE trace_id IN ({marks})", stale)
            conn.execute(f"DELETE FROM decision WHERE trace_id IN ({marks})",
                         stale)
            conn.execute(f"DELETE FROM trace WHERE trace_id IN ({marks})", stale)
        return {"traces": len(stale), "spans": spans, "decisions": decisions}

    def purge_tenant(self, tenant: str) -> Dict[str, int]:
        """Erase one tenant's telemetry entirely.

        Separate from time-based expiry because the request is different: a
        deletion request is not a retention policy, and it must not wait for
        one to come round."""
        with self._conn() as conn:
            traces = [r["trace_id"] for r in conn.execute(
                "SELECT trace_id FROM trace WHERE tenant = ?",
                (tenant,)).fetchall()]
            if not traces:
                return {"traces": 0, "spans": 0, "decisions": 0}
            marks = ",".join("?" * len(traces))
            spans = conn.execute(
                f"SELECT COUNT(*) AS n FROM span WHERE trace_id IN ({marks})",
                traces).fetchone()["n"]
            decisions = conn.execute(
                f"SELECT COUNT(*) AS n FROM decision WHERE trace_id IN ({marks})",
                traces).fetchone()["n"]
            conn.execute(f"DELETE FROM span WHERE trace_id IN ({marks})", traces)
            conn.execute(f"DELETE FROM decision WHERE trace_id IN ({marks})",
                         traces)
            conn.execute("DELETE FROM trace WHERE tenant = ?", (tenant,))
        return {"traces": len(traces), "spans": spans, "decisions": decisions}
