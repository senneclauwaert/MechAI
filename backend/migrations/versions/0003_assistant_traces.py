"""assistant traces

Creates the assistant schema and its traces table (one row per question asked).

Revision ID: 0003
Revises: 61783635aa12
Create Date: 2026-11-24 20:50:20.824877
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "61783635aa12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    """A JSONB type that stores None as SQL NULL, same as the model."""
    return postgresql.JSONB(none_as_null=True, astext_type=sa.Text())


# Create the schema first, then the table and its index on created_at.
def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS assistant")
    op.create_table(
        "traces",
        sa.Column("question", sa.String(length=500), nullable=False),
        sa.Column("kart_number", sa.String(length=10), nullable=True),
        sa.Column("input", sa.String(length=10), nullable=False),
        sa.Column("question_language", sa.String(length=5), nullable=True),
        sa.Column("safety_level", sa.String(length=20), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("search_terms", _jsonb(), nullable=True),
        sa.Column("retrieved", _jsonb(), nullable=False),
        sa.Column("sections_sent", _jsonb(), nullable=False),
        sa.Column("best_similarity", sa.Float(), nullable=True),
        sa.Column("prompt_versions", _jsonb(), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("document_revisions", _jsonb(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("latency_ms", _jsonb(), nullable=False),
        sa.Column("answer", _jsonb(), nullable=True),
        sa.Column("flags", _jsonb(), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("request_id", sa.String(length=20), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="assistant",
    )
    op.create_index("traces_created_at", "traces", ["created_at"], schema="assistant")


# Undo the upgrade: the table (and its index), then the schema.
def downgrade() -> None:
    op.drop_index("traces_created_at", table_name="traces", schema="assistant")
    op.drop_table("traces", schema="assistant")
    op.execute("DROP SCHEMA IF EXISTS assistant")
