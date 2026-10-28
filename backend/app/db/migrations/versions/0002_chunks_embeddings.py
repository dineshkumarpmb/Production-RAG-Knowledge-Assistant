"""Chunks, embedding cache and near-duplicate tracking.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
EMBEDDING_DIM = 1024


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("simhash", sa.BigInteger(), nullable=True))
        batch.add_column(sa.Column("near_duplicate_of_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_documents_near_duplicate_of_id_documents",
            "documents",
            ["near_duplicate_of_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("heading_path", JSONType, nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_chunks_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"),
    )
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])
    op.create_index("ix_chunks_text_hash", "chunks", ["text_hash"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "embedding_cache",
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("text_hash", "model", name="pk_embedding_cache"),
    )


def downgrade() -> None:
    op.drop_table("embedding_cache")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_text_hash", table_name="chunks")
    op.drop_index("ix_chunks_tenant_id", table_name="chunks")
    op.drop_table("chunks")
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("fk_documents_near_duplicate_of_id_documents", type_="foreignkey")
        batch.drop_column("near_duplicate_of_id")
        batch.drop_column("simhash")
