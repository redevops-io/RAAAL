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

from .retention import (
    RETENTION_POLICY_VERSION,
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
    payload["counts"] = {name: len(rows)
                         for name, rows in payload["tables"].items()}
    return payload


def delete_workspace(store, owner: str, *, requested_at: str,
                     request_id: Optional[str] = None) -> DeletionReceipt:
    """Remove one user's records, then prove none remain.

    Dependents before roots, because `plan_run` is found through `plan` and
    deleting the plan first would orphan the runs beyond reach.
    """
    request_id = request_id or f"del-{uuid.uuid4().hex[:16]}"
    counts: Dict[str, int] = {}

    ordered = sorted(owner_scoped_tables(),
                     key=lambda one: 0 if one.owner_scope is OwnerScope.INDIRECT
                     else 1)

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

    Reads the classified inventory directly rather than trusting the deletion
    that just ran. Verification derived from the deletion code would agree with
    it by construction.
    """
    remaining: Dict[str, int] = {}
    with store._conn() as conn:
        for record in owner_scoped_tables():
            rows = _rows_for(conn, record, owner)
            if rows:
                remaining[record.table] = len(rows)
    return remaining
