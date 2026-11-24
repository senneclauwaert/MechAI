"""document hash and ingest job

Adds file_hash (stops duplicate uploads) and job_id (last ingestion job) to documents,
and makes a page number unique within its document.

Revision ID: 61783635aa12
Revises: 0001
Create Date: 2026-11-20 00:11:37.121286
"""

from collections.abc import Sequence

import pgvector.sqlalchemy  # noqa: F401
import sqlalchemy as sa
from alembic import op

revision: str = "61783635aa12"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Add the two columns to documents, then their constraints.
def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("file_hash", sa.String(length=64), nullable=True), schema="knowledge"
    )
    op.add_column("documents", sa.Column("job_id", sa.Uuid(), nullable=True), schema="knowledge")
    op.create_unique_constraint(
        "documents_file_hash_key", "documents", ["file_hash"], schema="knowledge"
    )
    op.create_foreign_key(
        "documents_job_id_fkey",
        "documents",
        "jobs",
        ["job_id"],
        ["id"],
        source_schema="knowledge",
        referent_schema="core",
        ondelete="SET NULL",
    )
    op.create_unique_constraint(
        "pages_document_page", "pages", ["document_id", "page_number"], schema="knowledge"
    )


# Undo the upgrade, in reverse order.
def downgrade() -> None:
    op.drop_constraint("pages_document_page", "pages", schema="knowledge", type_="unique")
    op.drop_constraint("documents_job_id_fkey", "documents", schema="knowledge", type_="foreignkey")
    op.drop_constraint("documents_file_hash_key", "documents", schema="knowledge", type_="unique")
    op.drop_column("documents", "job_id", schema="knowledge")
    op.drop_column("documents", "file_hash", schema="knowledge")
