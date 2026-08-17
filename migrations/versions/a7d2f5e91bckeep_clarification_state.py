"""Persist clarification state, so answering is a write and looking is a read.

`POST /pilot/answer` rendered its result at the POST URL. Nothing was stored,
so the answers existed only in the request body: a refresh, a Back, or a pasted
link issued a GET against a POST-only route and got `Method Not Allowed`, and
Back returned to the last real GET — the empty form — discarding everything the
person had typed.

This is the table the redirect target reads. The POST persists and returns 303;
the GET loads this row and renders it. That makes reopening a clarification the
same kind of operation as reopening a saved plan: a read of a pinned artifact,
never a re-interpretation of the sentence.

Separate from `pilot_plans` deliberately. A saved plan is something a person
chose to keep. A review is where they had got to when they pressed the button.
Merging them would list every half-answered attempt under "Your plans", and
would give one retention policy to two things that deserve different ones.

Keyed `(owner, review_id)`. Tenant-owned, so the owner is part of the identity:
the id is content-addressed, and without the owner in the key two participants
who typed the same thing would collide and one would upsert over the other.

Revision ID: a7d2f5e91bc4
Revises: f4b81e7c9a26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'a7d2f5e91bc4'
down_revision: Union[str, Sequence[str], None] = 'f4b81e7c9a26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "pilot_reviews"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column("owner", sa.Text(), primary_key=True, nullable=False),
        sa.Column("review_id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("artifact", sa.Text(), nullable=False),
    )
    # Listing a participant's own reviews is the only query besides the
    # by-id read, and it is the one that runs on every page that offers to
    # resume. Indexed rather than left to a scan that is fine at pilot size
    # and stops being fine silently.
    op.create_index(f"ix_{TABLE}_owner_created", TABLE,
                    ["owner", "created_at"])


def downgrade() -> None:
    op.drop_index(f"ix_{TABLE}_owner_created", table_name=TABLE)
    op.drop_table(TABLE)
