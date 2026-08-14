"""Declare the runtime plan table that the application was creating itself.

`pilot_plans` existed in production and in no migration. The store created it
with `CREATE TABLE IF NOT EXISTS` the first time somebody saved a runtime plan,
which meant the table appeared partway through the life of a deployment, long
after the deploy that shipped the code.

Nothing noticed, because a running process does not re-run its preflight. The
schema-parity check refuses to serve against a database it cannot account for —
which is the behaviour that catches a half-applied migration — and it was
perfectly happy at startup, because at startup the table did not exist yet. The
refusal arrived at the *next* restart, on an unrelated deploy, reported as
`SCHEMA_MISMATCH ('remove_table', pilot_plans)`.

So the fault was never in the deploy that surfaced it. A table created on first
use is a landmine whose fuse is however long it takes somebody to restart.

`CREATE TABLE IF NOT EXISTS` here too, deliberately: every deployment that has
served a runtime plan already has this table, and it must not be an error to
find it. What changes is that it is now *declared* — present in the model, and
created by a migration on a database that has never seen one.

The downgrade drops it. It holds runtime plans, so that is a real deletion —
but a downgrade past the revision that introduced a table is a request to
return to a schema that has no such table, and leaving it behind would leave
the parity check failing in the other direction.
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'b8e4d1c05a97'
down_revision: Union[str, Sequence[str], None] = 'a91c4e7b2f05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: All four runtime tables, not just the one that fired.
#:
#: `pilot_plans` is the table that took the deployment down, because it is the
#: one that had been used. The other three are created by their own modules on
#: first use in exactly the same way, and had simply not been reached yet on
#: this deployment. Declaring only the one that failed would leave three
#: identical faults waiting for a participant to consent, an event to be
#: recorded, or a transcript to be written.
TABLES = (
    """
    CREATE TABLE IF NOT EXISTS pilot_plans (
        plan_id       TEXT NOT NULL,
        owner         TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        text          TEXT NOT NULL,
        artifact      TEXT NOT NULL,
        PRIMARY KEY (owner, plan_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pilot_consent (
        participant     TEXT PRIMARY KEY NOT NULL,
        state           TEXT NOT NULL,
        at              TEXT NOT NULL,
        notice_version  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pilot_events (
        event_id      TEXT PRIMARY KEY NOT NULL,
        at            TEXT NOT NULL,
        kind          TEXT NOT NULL,
        plan_id       TEXT,
        participant   TEXT,
        detail        TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pilot_transcripts (
        entry_id      TEXT PRIMARY KEY NOT NULL,
        participant   TEXT NOT NULL,
        at            TEXT NOT NULL,
        attempt       INTEGER NOT NULL,
        text          TEXT NOT NULL,
        detail        TEXT NOT NULL
    )
    """,
)

DROPPED = ("pilot_transcripts", "pilot_events", "pilot_consent", "pilot_plans")


#: What a runtime plan saved before this migration belongs to.
#:
#: The runtime pilot served one participant, so every existing row is theirs.
#: Backfilled rather than left null so erasure reaches these rows by the same
#: declared owner path as every other personal record — an unscoped table
#: cannot be deleted with its owner, and leaving rows behind while reporting a
#: completed erasure is the failure that classification exists to prevent.
PILOT_OWNER = "pilot"


def upgrade() -> None:
    # `PRIMARY KEY NOT NULL` rather than `PRIMARY KEY` alone. SQLite permits a
    # null in a TEXT primary key — a documented quirk it keeps for backward
    # compatibility — so without the redundant constraint the migrated schema
    # and the model disagree about nullability and `test_migration_parity`
    # fails. On PostgreSQL a primary key already implies it, which is why
    # production never showed this and a fresh SQLite database does.
    #
    # Raw SQL rather than `op.create_table`, so the IF NOT EXISTS is
    # expressible. The column types match what the modules have been creating
    # since the runtime shipped; changing them here would rewrite data that
    # already exists under a definition nobody chose.
    for statement in TABLES:
        op.execute(statement)

    # The column, on a database that already has the table.
    #
    # Checked through the inspector rather than `ADD COLUMN IF NOT EXISTS`,
    # which PostgreSQL has and SQLite does not — and this migration has to run
    # on both, because the parity test builds a fresh SQLite database from it.
    import sqlalchemy as sa

    bind = op.get_bind()
    columns = {column["name"]
               for column in sa.inspect(bind).get_columns("pilot_plans")}
    if "owner" not in columns:
        op.execute("ALTER TABLE pilot_plans ADD COLUMN owner TEXT")

    op.execute(
        f"UPDATE pilot_plans SET owner = '{PILOT_OWNER}' WHERE owner IS NULL")

    # Re-key an existing table onto (plan_id, owner).
    #
    # A table created before this migration is keyed on plan_id alone. The
    # standing rule is that a table with an owner keys by it — otherwise two
    # tenants cannot hold the same id, and the store's delete-then-insert lets
    # one overwrite the other. PostgreSQL can move a primary key in place;
    # SQLite cannot, and does not need to, because there the table was created
    # by the statement above with the key already right.
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE pilot_plans ALTER COLUMN owner SET NOT NULL")
        op.execute("ALTER TABLE pilot_plans DROP CONSTRAINT IF EXISTS "
                   "pilot_plans_pkey")
        op.execute("ALTER TABLE pilot_plans ADD PRIMARY KEY (owner, plan_id)")


def downgrade() -> None:
    for table in DROPPED:
        op.execute(f"DROP TABLE IF EXISTS {table}")
