"""invalidate runs whose declared rule was never executed

Revision ID: f2b7c91d40ae
Revises: c4a1e93b7f28
Create Date: 2026-08-05

A pilot user described "buy $1,000 of an S&P 500 ETF every time the index
crosses below its 200-day moving average" and received a figure. The figure was
for buying once and holding: `_run` called `simulate(..., program=buy_and_hold(
tradeable))` regardless of what the scenario declared, and nothing ever
converted `event_program` into an `EventProgram`. The result equalled the
buy-and-hold benchmark exactly, beside a disclosure saying the difference was
attributable to the rule.

`run_invalidation` records that a stored run must not be read as a strategy
result.

**The run is kept.** Deleting it would destroy the evidence that the defect
happened and what was shown, and the correction is not that the number was
mistyped — it is that the number answers a different question than the plan
asked. A user who remembers seeing $5,160 must be able to find the record of
having been shown it.

**No foreign key to `plan_run`.** A future decision to purge runs on a
retention schedule must not be blocked by, or silently delete, the record that
one of them was wrong. The pairing is by id and is allowed to outlive the row.

**Nothing is back-filled by this migration.** Which runs are affected is
derived from the stored artifacts by `src.workspace.invalidate`, because the
predicate is a property of each scenario and result rather than of a date, and
a migration that guessed would produce exactly the manufactured evidence the
market-data migration refused to produce.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2b7c91d40ae'
down_revision: Union[str, Sequence[str], None] = 'c4a1e93b7f28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_invalidation",
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("plan_id", sa.Text(), nullable=False),
        # The runtime's own vocabulary — RULE_NOT_EXECUTED — not prose. This is
        # the column an operator groups by when asking how far a defect reached.
        sa.Column("classification", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        # Which engine produced the invalid run, so a replacement run can state
        # what changed rather than merely being newer.
        sa.Column("engine_version", sa.Text(), nullable=False),
        sa.Column("invalidated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("owner", "run_id"),
    )
    op.create_index("run_invalidation_plan", "run_invalidation",
                    ["owner", "plan_id"])


def downgrade() -> None:
    op.drop_index("run_invalidation_plan", table_name="run_invalidation")
    op.drop_table("run_invalidation")
