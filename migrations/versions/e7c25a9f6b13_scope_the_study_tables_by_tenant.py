"""Give the study tables a tenant, without giving them a subject.

`pilot_consent`, `pilot_events` and `pilot_transcripts` were classified as
tenant-owned and carried no `owner`, so the tenancy invariant failed the moment
they were declared. Two registries disagreed about what "owned" meant: the
retention registry recorded `owner_column="participant"`, and the participant is
not a tenant.

**The fix separates the two namespaces rather than merging them.**

    owner        the tenant. Decides who may read the row. In the primary key,
                 as on every other tenant-owned table, so one tenant's write
                 cannot replace another's.
    participant  the study pseudonym. Says which subject produced the row. Not
                 a foreign key to any authenticated user, and not part of the
                 key on events or transcripts.

The tempting shortcut was to key events and transcripts by `(participant, id)`.
That would have made the study pseudonym part of *storage identity*, and
deduplication, foreign keys, exports, restores and every future migration would
then depend on the participant↔user association — the one link that has to stay
severable. Keeping it out of the key is what lets that association be deleted
later without destroying or re-keying the experimental evidence.

`participant` is in the key on `pilot_consent` alone, because a consent record
*is* the statement that one participant agreed. It is intrinsic to what that row
is, where an event merely happens to have been produced by somebody.

`pilot_transcripts.participant` also becomes nullable, so a pseudonym can be
scrubbed from a transcript while the words survive.

Backfilled to the shared pilot owner, which is what every existing row belongs
to: these tables predate per-viewer ownership, so there is no other tenant they
could have come from and no guessing involved.

Revision ID: e7c25a9f6b13
Revises: d3f6a2b81c04
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'e7c25a9f6b13'
down_revision: Union[str, Sequence[str], None] = 'd3f6a2b81c04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: What every existing row belongs to. The tables were written before ownership
#: was a question, under a single shared workspace, so this is the fact rather
#: than a default.
SHARED = "pilot"

#: table -> the columns its key becomes.
REKEYED = {
    "pilot_consent": ("owner", "participant"),
    "pilot_events": ("owner", "event_id"),
    "pilot_transcripts": ("owner", "entry_id"),
}

#: The finished shape, for the dialect that cannot alter one.
#:
#: SQLite can add a column and nothing else: not a primary key, not a NOT NULL,
#: not a NULL. So there the table is rebuilt — create, copy, drop, rename —
#: which is the ordinary way and the only way. Spelled out rather than derived
#: from `schema.py`, because a migration that imported the model would agree
#: with it by construction and `test_migration_parity` exists to find out
#: whether they agree in fact.
REBUILT = {
    "pilot_consent": """
        CREATE TABLE pilot_consent__new (
            owner           TEXT NOT NULL,
            participant     TEXT NOT NULL,
            state           TEXT NOT NULL,
            at              TEXT NOT NULL,
            notice_version  TEXT NOT NULL,
            PRIMARY KEY (owner, participant)
        )""",
    "pilot_events": """
        CREATE TABLE pilot_events__new (
            owner         TEXT NOT NULL,
            event_id      TEXT NOT NULL,
            at            TEXT NOT NULL,
            kind          TEXT NOT NULL,
            plan_id       TEXT,
            participant   TEXT,
            detail        TEXT NOT NULL,
            PRIMARY KEY (owner, event_id)
        )""",
    "pilot_transcripts": """
        CREATE TABLE pilot_transcripts__new (
            owner         TEXT NOT NULL,
            entry_id      TEXT NOT NULL,
            participant   TEXT,
            at            TEXT NOT NULL,
            attempt       INTEGER NOT NULL,
            text          TEXT NOT NULL,
            detail        TEXT NOT NULL,
            PRIMARY KEY (owner, entry_id)
        )""",
}

#: The columns carried across, in order. `owner` is supplied by the SELECT
#: rather than copied, because the old table may not have it at all.
CARRIED = {
    "pilot_consent": ("participant", "state", "at", "notice_version"),
    "pilot_events": ("event_id", "at", "kind", "plan_id", "participant",
                     "detail"),
    "pilot_transcripts": ("entry_id", "participant", "at", "attempt", "text",
                          "detail"),
}


def _columns(bind, table) -> set:
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    postgresql = bind.dialect.name == "postgresql"

    for table, key in REKEYED.items():
        columns = _columns(bind, table)
        if not columns:
            continue                          # nothing here to re-key

        if not postgresql:
            _rebuild(table, columns)
            continue

        if "owner" not in columns:
            op.execute(f"ALTER TABLE {table} ADD COLUMN owner TEXT")
        op.execute(
            f"UPDATE {table} SET owner = '{SHARED}' WHERE owner IS NULL")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN owner SET NOT NULL")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS "
                   f"{table}_pkey")
        op.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({', '.join(key)})")

    # A transcript may lose its pseudonym and keep its words. Dropping NOT NULL
    # is what makes scrubbing possible without deleting evidence, and it is the
    # half of this change that has nothing to do with tenancy. On SQLite the
    # rebuild above already produced it.
    if postgresql and "participant" in _columns(bind, "pilot_transcripts"):
        op.execute("ALTER TABLE pilot_transcripts "
                   "ALTER COLUMN participant DROP NOT NULL")


def _rebuild(table: str, columns: set) -> None:
    """Create, copy, drop, rename — the only way to re-key on SQLite.

    Rows are claimed for the shared workspace as they cross, from the old
    table's own `owner` where it somehow has one and from the constant where it
    does not. A row that arrived before ownership was a question belongs to the
    single workspace that existed then; leaving it NULL would fail the new NOT
    NULL and lose the study's history to a schema change.
    """
    carried = ", ".join(CARRIED[table])
    owner = "owner" if "owner" in columns else f"'{SHARED}'"
    op.execute(f"DROP TABLE IF EXISTS {table}__new")
    op.execute(REBUILT[table])
    op.execute(f"INSERT INTO {table}__new (owner, {carried}) "
               f"SELECT COALESCE({owner}, '{SHARED}'), {carried} FROM {table}")
    op.execute(f"DROP TABLE {table}")
    op.execute(f"ALTER TABLE {table}__new RENAME TO {table}")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, key in REKEYED.items():
        if "owner" not in _columns(bind, table):
            continue
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS "
                   f"{table}_pkey")
        op.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({key[-1]})")
        op.execute(f"ALTER TABLE {table} DROP COLUMN owner")
