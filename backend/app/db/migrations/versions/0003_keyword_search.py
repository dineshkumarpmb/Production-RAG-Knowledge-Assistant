"""Full-text search column for keyword retrieval (PostgreSQL only).

`chunks.tsv` is generated from the heading path (weight A) and the chunk text (weight B), so a
query matching a section title ranks above one matching only body text. It uses the 'english'
configuration (stemming + stop words); per-language configurations can come later.

The column is not mapped in the ORM models; keyword search reads it through SQL expressions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return  # SQLite (unit tests) has no full-text search; retrieval tests run on Postgres
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(heading_path::text, '')), 'A')
            || setweight(to_tsvector('english', text), 'B')
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsv")
