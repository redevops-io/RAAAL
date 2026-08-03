"""An indirectly owned table that exists only for tests.

`plan_run` used to be the only indirectly scoped table, and it was doing double
duty: a real domain record *and* the proof that `OwnerScope.INDIRECT` and
`OwnershipPath` worked. That made the production schema weaker than it needed to
be — every ownership question was a join, run ids were not tenant-safe, and the
deletion path depended on a cascade nobody enforced.

Once it was given its own owner, nothing was indirect any more, and the
machinery for indirect ownership would have gone untested while remaining in the
codebase for the next table that needs it. Keeping a domain table in a weaker
shape to serve as a canary is paying for the test in production; this pays for
it here instead.

The table is created and dropped by the fixture, and is never in the migrations
or the production registry — `tests/test_retention.py` reads the schema and
fails on an unclassified table, so a stray one would be caught.
"""
from __future__ import annotations

from src.workspace.retention import (
    DataClass,
    DeletionBehaviour,
    OwnerScope,
    OwnershipPath,
    RecordClass,
)

TABLE = "ownership_test_child"

#: Reachable only through `worksheet_proposal`, exactly as `plan_run` was
#: reachable only through `plan`.
INDIRECT_CHILD = RecordClass(
    table=TABLE,
    data_class=DataClass.PERSONAL_RECORD,
    owner_scope=OwnerScope.INDIRECT,
    # Spans the parent's whole key. `proposal_id` alone matches both tenants'
    # proposals, and deleting one owner then removed the other's child rows —
    # with correct-looking counts, because the query was valid and returned
    # rows. A child of a tenant-owned parent has to carry the tenant dimension;
    # that is the same invariant as the keys themselves, one level down.
    ownership_path=OwnershipPath(
        local_key=("proposal_id", "proposal_owner"),
        parent_table="worksheet_proposal",
        parent_key=("proposal_id", "owner"), parent_owner_column="owner"),
    retention_policy="retained while the account is active",
    deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
    export_behaviour="included in a workspace export",
    contains_sensitive_financial_data=False,
    contains_model_content=False,
)

CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    child_id       TEXT NOT NULL,
    proposal_id    TEXT NOT NULL,
    proposal_owner TEXT NOT NULL,
    note           TEXT NOT NULL,
    PRIMARY KEY (child_id)
)
"""

DROP = f"DROP TABLE IF EXISTS {TABLE}"


def create(store) -> None:
    with store._conn() as conn:
        conn.execute(CREATE)


def drop(store) -> None:
    with store._conn() as conn:
        conn.execute(DROP)


def add(store, *, child_id: str, proposal_id: str, proposal_owner: str,
        note: str = "n") -> None:
    with store._conn() as conn:
        conn.execute(
            f"INSERT INTO {TABLE} (child_id, proposal_id, proposal_owner, note) "
            "VALUES (?,?,?,?)",
            (child_id, proposal_id, proposal_owner, note))


def rows(store):
    with store._conn() as conn:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE}")]
