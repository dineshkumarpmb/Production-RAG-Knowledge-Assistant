"""Initial schema: tenants, users, documents, document pages.

Revision ID: 0001
Revises:
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        # Not used until Phase 2, but enabling it now fails fast if the image lacks pgvector.
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("name", name="uq_tenants_name"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("role IN ('admin', 'editor', 'viewer')", name="ck_users_role_valid"),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_users_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_uri", sa.String(1024), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("author", sa.String(500), nullable=True),
        sa.Column("created_at_src", sa.DateTime(timezone=True), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(16), nullable=True),
        sa.Column("metadata", JSONType, nullable=False),
        sa.Column("visibility", sa.String(10), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'parsing', 'embedding', 'ready', 'failed', 'duplicate')",
            name="ck_documents_status_valid",
        ),
        sa.CheckConstraint(
            "visibility IN ('private', 'group', 'tenant')", name="ck_documents_visibility_valid"
        ),
        sa.CheckConstraint(
            "file_type IN ('pdf', 'docx', 'txt', 'md')", name="ck_documents_file_type_valid"
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="fk_documents_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_documents_tenant_id_tenants",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("tenant_id", "sha256", name="uq_documents_tenant_id_sha256"),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.create_index("ix_documents_tenant_id_status", "documents", ["tenant_id", "status"])

    op.create_table(
        "document_pages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("is_low_text", sa.Boolean(), nullable=False),
        sa.Column("headings", JSONType, nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_pages_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_pages"),
        sa.UniqueConstraint(
            "document_id", "page_number", name="uq_document_pages_document_id_page_number"
        ),
    )


def downgrade() -> None:
    op.drop_table("document_pages")
    op.drop_index("ix_documents_tenant_id_status", table_name="documents")
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_users_tenant_id", table_name="users")
    op.drop_table("users")
    op.drop_table("tenants")
