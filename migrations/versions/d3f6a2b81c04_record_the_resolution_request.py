"""Record which request produced a delivery, not only which snapshot.

The delivery record named the snapshot, the columns, the row count and the
digest — and not what was asked for. That is enough to say *what was received*
and not enough to *check it*, because the frame depends on the request as well
as the snapshot: resolving the same snapshot with dividends reinvested returns
a different frame with a different digest.

The consequence was already sitting in the suite, mislabelled. The provenance
journey recomputed a digest without the request, compared a price-return frame
against a total-return one, and failed — which read as "resolution is not
deterministic" and would have condemned the snapshot-by-hash design it is meant
to support. Resolution was deterministic. The record was incomplete.

Nullable, and deliberately not backfilled. An event written before this column
existed genuinely does not say whether dividends were reinvested, and writing
today's default into it would manufacture an answer — letting a delivery be
"verified" against a frame it may never have carried. Absent stays absent, and
`MarketDataAccessEvent.reproducible` reports those events as what they are:
coherent, and not checkable against the data.

Revision ID: d3f6a2b81c04
Revises: b8e4d1c05a97
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from src.db.types import JsonText

revision: str = 'd3f6a2b81c04'
down_revision: Union[str, Sequence[str], None] = 'b8e4d1c05a97'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "market_data_access_event"
COLUMN = "resolution"


def _has_column(bind) -> bool:
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return True                    # nothing to alter; schema.py declares it
    return COLUMN in {c["name"] for c in inspector.get_columns(TABLE)}


def upgrade() -> None:
    """Add the column where it is missing.

    Guarded rather than unconditional: a database created from `schema.py`
    already has it, and a migration that assumed otherwise would fail on a
    fresh deployment while succeeding on an upgraded one — which is the pair of
    outcomes that makes a migration untestable.
    """
    bind = op.get_bind()
    if _has_column(bind):
        return
    op.add_column(TABLE, sa.Column(COLUMN, JsonText(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if not _has_column(bind):
        return
    op.drop_column(TABLE, COLUMN)
