"""The snapshot descriptor index: shared reference data, no tenant column.

Keyed by `descriptor_hash`, with `snapshot_hash` indexed rather than unique.
The same observation bytes can legitimately carry several descriptions over
time — a licence re-reviewed, an adapter version corrected, provenance filled
in — and each is a distinct record of what was believed about that data. The
observations' identity stays fixed while the description moves.

Carries no `owner`. It is the first `SHARED_REFERENCE` table: a market snapshot
describes the world rather than a user, and the rule for such a table is
stricter than the tenant rule rather than an exemption from it — no
tenant-identifying column at all. A meaningless `owner` added to satisfy a
check is a column somebody will scope a query by and somebody else will trust.

Revision ID: f4b81e7c9a26
Revises: e7c25a9f6b13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from src.db.types import JsonText

revision: str = 'f4b81e7c9a26'
down_revision: Union[str, Sequence[str], None] = 'e7c25a9f6b13'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "market_snapshot"


def _exists(bind) -> bool:
    return TABLE in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if _exists(bind):
        return
    op.create_table(
        TABLE,
        sa.Column("descriptor_hash", sa.Text(), primary_key=True),
        sa.Column("snapshot_hash", sa.Text(), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("symbols", JsonText(), nullable=False),
        sa.Column("range_start", sa.Text(), nullable=False),
        sa.Column("range_end", sa.Text(), nullable=False),
        sa.Column("sessions", sa.Integer(), nullable=False),
        sa.Column("resolution", JsonText(), nullable=False),
        sa.Column("corporate_actions", sa.Text(), nullable=False),
        sa.Column("calendar", sa.Text(), nullable=False),
        sa.Column("source_adapter", sa.Text(), nullable=False),
        sa.Column("source_adapter_version", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("data_as_of", sa.Text(), nullable=False),
        sa.Column("license_class", sa.Text(), nullable=False),
        sa.Column("license_review_status", sa.Text(), nullable=False),
        sa.Column("content_digest_version", sa.Text(), nullable=False),
        sa.Column("contract_version", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.Text(), nullable=False),
    )
    op.create_index("market_snapshot_by_content", TABLE, ["snapshot_hash"])


def downgrade() -> None:
    bind = op.get_bind()
    if not _exists(bind):
        return
    op.drop_index("market_snapshot_by_content", table_name=TABLE)
    op.drop_table(TABLE)
