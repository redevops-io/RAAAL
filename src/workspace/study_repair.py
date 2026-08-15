"""Give an already-created study table the tenant column it now needs.

`pilot_events` had this for one column and now three tables need it for the
same one, which is when a copied fragment earns a name.

**Not a migration in `db.migrate`'s sense, and deliberately so.** These tables
are study instrumentation: they hold counts, consent and prose, nothing
financial, and losing one costs a measurement rather than a plan. `db.migrate`
owns the schema that holds people's money and refuses to start when it is
wrong; this repairs a table the application creates for itself with `CREATE
TABLE IF NOT EXISTS`, which says nothing about columns and therefore leaves an
older table exactly as it found it.

What it must not do is fail quietly. A stale table plus `record`'s exception
guard is the combination that reports a pilot nobody used, and that is the
failure this exists to prevent rather than cause.

**Existing rows are backfilled, not left NULL.** A row written before the
column existed belongs to the shared pilot workspace — these tables predate
per-viewer ownership, so there is no other tenant it could have come from. Left
NULL it would simply stop being found, and a study that silently drops its own
history is worse than one that fails loudly.
"""
from __future__ import annotations

from typing import Sequence

#: What rows written before ownership was a question belong to. The same value
#: `owner.SHARED` uses, and the same one the migration backfills, because a
#: disagreement between them would split one workspace's history in two.
SHARED = "pilot"


def _columns(connection, table: str) -> Sequence[str]:
    """Column names, or nothing when the question cannot be asked here.

    `PRAGMA table_info` is SQLite's. On PostgreSQL this raises, and the answer
    is to do nothing: there the schema is owned by `db.migrate`, which has
    already added the column and moved the key — work this could not do anyway,
    since SQLite cannot alter a primary key in place.
    """
    try:
        return [row[1] for row in
                connection.execute(f"PRAGMA table_info({table})")]
    except Exception:                                          # noqa: BLE001
        return ()


def ensure_owner(connection, table: str) -> None:
    """Add `owner` where an older table lacks it, and claim its rows."""
    present = _columns(connection, table)
    if not present or "owner" in present:
        return
    try:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN owner TEXT")
        connection.execute(
            f"UPDATE {table} SET owner = ? WHERE owner IS NULL", (SHARED,))
        connection.commit()
    except Exception:                                          # noqa: BLE001
        return
