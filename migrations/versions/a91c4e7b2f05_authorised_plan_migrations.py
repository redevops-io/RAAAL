"""record an authorised recompilation of a saved plan

Revision ID: a91c4e7b2f05
Revises: f2b7c91d40ae
Create Date: 2026-08-06

A plan saved before the funding policy existed rebuilds as not-event-funded,
so the engine that now executes conditional rules refuses it exactly as the one
that could not. The only way to produce a corrected result is to recompile from
the pinned parse under a newer compiler — which changes what a saved plan
means, and `migration_for` has always said only its owner may agree to that.

`plan_migration` records that agreement, and holds the interpretation it
authorised.

**The scenario lives here rather than in `plan`.** `plan.scenario` is immutable
and keyed on the plan id, correctly: the pinned parse is the thing the user
read and confirmed. Overwriting it would erase the interpretation they agreed
to in order to store one they agreed to later, and the withdrawn run would then
be attached to a scenario that no longer explains it.

**`new_run` is nullable.** The record is created before the run it names —
the alternative is a run citing a migration that does not exist yet, and the
gap between the two is exactly when a crash happens.

**No foreign keys to `plan_run`.** A retention policy that later purges runs
must not be able to delete the record of why a result was replaced, nor be
blocked by it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a91c4e7b2f05'
down_revision: Union[str, Sequence[str], None] = 'f2b7c91d40ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "plan_migration",
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("migration_id", sa.Text(), nullable=False),
        sa.Column("plan_id", sa.Text(), nullable=False),
        sa.Column("from_compiler", sa.Text(), nullable=False),
        sa.Column("to_compiler", sa.Text(), nullable=False),
        sa.Column("from_engine", sa.Text(), nullable=False),
        sa.Column("to_engine", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("authorized_by", sa.Text(), nullable=False),
        sa.Column("migrated_at", sa.Text(), nullable=False),
        sa.Column("old_run", sa.Text(), nullable=True),
        sa.Column("new_run", sa.Text(), nullable=True),
        # JSONB on PostgreSQL, TEXT on SQLite — the same variant the model
        # declares. Written as plain TEXT here, the migration produced a schema
        # the parity check refused: a column that stores JSON as text cannot be
        # queried as JSON, and the disagreement would only surface the first
        # time someone tried.
        sa.Column("scenario",
                  sa.Text().with_variant(
                      postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                  nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("owner", "migration_id"),
    )
    op.create_index("plan_migration_plan", "plan_migration",
                    ["owner", "plan_id"])


def downgrade() -> None:
    op.drop_index("plan_migration_plan", table_name="plan_migration")
    op.drop_table("plan_migration")
