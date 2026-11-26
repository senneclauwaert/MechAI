"""trace raw replies

Adds raw_replies to assistant.traces: the model's text when it could not be used.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-24 22:40:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Existing traces get an empty list.
def upgrade() -> None:
    op.add_column(
        "traces",
        sa.Column(
            "raw_replies",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        schema="assistant",
    )


# Undo the upgrade.
def downgrade() -> None:
    op.drop_column("traces", "raw_replies", schema="assistant")
