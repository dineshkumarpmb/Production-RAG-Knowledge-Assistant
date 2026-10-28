"""Conversation memory and query-log details: running summaries, rewritten queries, request IDs.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("conversations") as batch:
        batch.add_column(sa.Column("summary", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("summarized_messages", sa.Integer(), server_default="0", nullable=False)
        )
    with op.batch_alter_table("query_logs") as batch:
        batch.add_column(sa.Column("rewritten_query", sa.Text(), nullable=True))
        batch.add_column(sa.Column("request_id", sa.String(64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("query_logs") as batch:
        batch.drop_column("request_id")
        batch.drop_column("rewritten_query")
    with op.batch_alter_table("conversations") as batch:
        batch.drop_column("summarized_messages")
        batch.drop_column("summary")
