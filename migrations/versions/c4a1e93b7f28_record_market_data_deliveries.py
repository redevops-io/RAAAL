"""record market data deliveries

Revision ID: c4a1e93b7f28
Revises: 7d37e87ad210
Create Date: 2026-08-03

A stored run cited what its producer *declared* it had used. The producer is
the one component whose claim is not independent evidence — it is exactly what
a defect would corrupt — and this run path has already been caught dropping the
resolver's answer while looking entirely correct.

`market_data_access_event` records the delivery itself: which request, which
run, which provenance record, and the digest of the exact canonical frame that
was handed over. `plan_run.access_event_id` cites it.

**The column is added nullable and stays nullable.** Runs recorded before this
existed have no delivery record, and back-filling one from today's
configuration would manufacture the very evidence the table exists to provide —
the same reason `MARKET_DATA_PROVENANCE_NOT_RECORDED` is a status rather than a
blank. A live producer may not leave it null; that is enforced in `generate`,
which is the layer that can tell "historical absence" from "this code declined
to record it".

**The foreign key points one way only.** The event is written before the run it
names, so `market_data_access_event.run_id` cannot carry a constraint — the run
does not exist yet. RESTRICT on the other direction is what matters: while a
run exists, the delivery it cites cannot be deleted, so a stored figure cannot
be made unverifiable by a deletion somewhere else.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c4a1e93b7f28'
down_revision: Union[str, Sequence[str], None] = '7d37e87ad210'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_data_access_event",
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("access_event_id", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("snapshot_id", sa.Text(), nullable=True),
        sa.Column("provenance_digest", sa.Text(), nullable=False),
        sa.Column("frame_digest", sa.Text(), nullable=False),
        sa.Column("selected_columns",
                  sa.Text().with_variant(
                      postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                  nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("range_start", sa.Text(), nullable=True),
        sa.Column("range_end", sa.Text(), nullable=True),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("access_decision", sa.Text(), nullable=False),
        sa.Column("accessed_at", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("owner", "access_event_id"),
    )
    op.create_index("access_event_run", "market_data_access_event",
                    ["owner", "run_id"])

    with op.batch_alter_table("plan_run") as batch:
        batch.add_column(sa.Column("access_event_id", sa.Text(), nullable=True))
        batch.create_foreign_key(
            "fk_plan_run_access_event", "market_data_access_event",
            ["owner", "access_event_id"], ["owner", "access_event_id"],
            ondelete="RESTRICT")


def downgrade() -> None:
    with op.batch_alter_table("plan_run") as batch:
        batch.drop_constraint("fk_plan_run_access_event", type_="foreignkey")
        batch.drop_column("access_event_id")
    op.drop_index("access_event_run", table_name="market_data_access_event")
    op.drop_table("market_data_access_event")
