"""Exporting and deleting one user's workspace, and proving it happened.

    enumerate owner records
        -> delete dependents, then roots
        -> write a receipt outside the deleted scope
        -> verify nothing remains

**Deletion is executed and then verified, not asserted.** A delete that silently
missed a table looks exactly like one that worked, so `verify_deleted` re-reads
every classified table and fails if a row survives. It reads the schema rather
than the deletion code, so the two cannot agree by construction.

**Indirect ownership is enumerated explicitly.** `plan_run` has no owner column
and is reachable only through its plan. A deletion written around
`WHERE owner = ?` removes nothing from it and reports success — which is the
shape of every cascade bug this file exists to prevent.

**The receipt holds no deleted content.** It records that a deletion happened,
what it touched and under which policy, using an irreversible reference to the
owner. A receipt reproducing the personal data it certifies as gone would be the
one surviving copy.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..db.schema import deletion_order
from . import retention
from .retention import (
    RETENTION_POLICY_VERSION,
    DeletionBehaviour,
    OwnerScope,
    owner_scoped_tables,
)


class DeletionIncomplete(RuntimeError):
    """Rows survived a deletion that reported success."""


@dataclass(frozen=True)
class DeletionReceipt:
    """Proof a deletion happened, holding none of what it deleted."""

    request_id: str
    owner_reference: str
    """An irreversible hash. A receipt naming the user would be a record of the
    person whose records were removed."""

    requested_at: str
    counts: Mapping[str, int]
    policy_version: str
    status: str
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"request_id": self.request_id,
                "owner_reference": self.owner_reference,
                "requested_at": self.requested_at, "counts": dict(self.counts),
                "policy_version": self.policy_version, "status": self.status,
                "detail": self.detail}


def owner_reference(owner: str) -> str:
    """A stable, irreversible reference for a receipt."""
    return "owner-" + hashlib.sha256(f"erasure:{owner}".encode()).hexdigest()[:32]


def _rows_for(conn, record, owner: str) -> List[Dict]:
    """Rows this owner holds, by the declared ownership path.

    No table is special-cased here. The join comes from the classification, so
    a new indirectly-owned table is reached correctly by declaring its path
    rather than by editing this function — which is the edit everyone forgets.
    """
    if record.owner_scope is OwnerScope.DIRECT:
        return [dict(r) for r in conn.execute(
            f"SELECT * FROM {record.table} WHERE {record.owner_column} = ?",
            (owner,))]
    if record.ownership_path is not None:
        return [dict(r) for r in conn.execute(
            record.ownership_path.select(record.table), (owner,))]
    raise DeletionIncomplete(
        f"{record.table} is indirectly scoped and declares no ownership path. "
        "Deleting around it would leave rows behind and report success")


def export_workspace(store, owner: str) -> Dict[str, Any]:
    """Everything one user's account holds, by table.

    Reads the same classified inventory deletion uses, so a table that would be
    deleted is a table that can be exported — an export missing something a
    deletion removes is a user who could not see what they lost.
    """
    payload: Dict[str, Any] = {"owner_reference": owner_reference(owner),
                               "policy_version": RETENTION_POLICY_VERSION,
                               "tables": {}}
    with store._conn() as conn:
        for record in owner_scoped_tables():
            payload["tables"][record.table] = _rows_for(conn, record, owner)
    _mark_withdrawn_runs(payload["tables"])
    payload["counts"] = {name: len(rows)
                         for name, rows in payload["tables"].items()}
    return payload


def _mark_withdrawn_runs(tables: Dict[str, Any]) -> None:
    """Attach each withdrawal to the run it withdraws.

    `run_invalidation` is exported as its own table because the inventory says
    so, and that alone is not enough: a consumer reading `plan_run` sees a
    figure with nothing on it, and would have to know to join a second table
    before believing the number. A reader who must remember to check is a
    reader who will quote the figure — which is precisely how a withdrawn
    result regains authority by leaving the interface that withdrew it.

    The same fix as `WorkspaceStore.runs_for`. Marked rather than removed: the
    user is entitled to everything their account holds, including the record
    that they were once shown a figure and what was wrong with it.
    """
    withdrawals = {row.get("run_id"): row
                   for row in tables.get("run_invalidation") or ()}
    if not withdrawals:
        return
    for row in tables.get("plan_run") or ():
        withdrawal = withdrawals.get(row.get("run_id"))
        if withdrawal is not None:
            row["invalidation"] = withdrawal


def delete_workspace(store, owner: str, *, requested_at: str,
                     request_id: Optional[str] = None) -> DeletionReceipt:
    """Remove one user's records, then prove none remain.

    Dependents before roots, because `plan_run` is found through `plan` and
    deleting the plan first would orphan the runs beyond reach.
    """
    request_id = request_id or f"del-{uuid.uuid4().hex[:16]}"
    counts: Dict[str, int] = {}

    # Derived from the relationship graph, not from a heuristic. The previous
    # ordering put indirectly-owned tables first, which was right for the one
    # indirect table that existed and says nothing about a second — and says
    # nothing at all about a dependency between two directly-owned tables, which
    # is what `event_reconciliation` referencing its events is.
    position = {name: index for index, name in enumerate(deletion_order())}
    # A classified table outside the relationship graph is deleted first.
    # Nothing declares a dependency on it, and deleting a dependent early can
    # never violate a RESTRICT — whereas guessing it is a parent could leave
    # its children unreachable.
    ordered = sorted(owner_scoped_tables(),
                     key=lambda one: position.get(one.table, -1))

    with store._conn() as conn:
        for record in ordered:
            rows = _rows_for(conn, record, owner)
            counts[record.table] = len(rows)
            if not rows:
                continue
            if record.owner_scope is OwnerScope.DIRECT:
                conn.execute(
                    f"DELETE FROM {record.table} "
                    f"WHERE {record.owner_column} = ?", (owner,))
            else:
                conn.execute(record.ownership_path.delete(record.table),
                             (owner,))

    remaining = verify_deleted(store, owner)
    if remaining:
        raise DeletionIncomplete(
            f"deletion for {owner_reference(owner)} left rows in "
            f"{', '.join(sorted(remaining))}. A deletion that reports success "
            "while records survive is worse than one that fails")

    return DeletionReceipt(
        request_id=request_id, owner_reference=owner_reference(owner),
        requested_at=requested_at, counts=counts,
        policy_version=RETENTION_POLICY_VERSION, status="COMPLETE",
        detail="every classified owner-scoped table was enumerated and emptied")


def verify_deleted(store, owner: str) -> Mapping[str, int]:
    """Tables still holding rows for this owner.

    Reads the registry itself, not the helper the deletion iterates. Both used
    to call `owner_scoped_tables()`, so a table missing from that list was
    skipped by the deletion *and* by the check that was supposed to catch the
    deletion skipping it — the two agreed by construction, which is precisely
    what this function's independence was meant to prevent.
    """
    remaining: Dict[str, int] = {}
    classified = [one for one in retention.WORKSPACE_RECORDS.values()
                  if one.deletion_behaviour is DeletionBehaviour.DELETE_WITH_OWNER]
    with store._conn() as conn:
        for record in classified:
            rows = _rows_for(conn, record, owner)
            if rows:
                remaining[record.table] = len(rows)
    return remaining
