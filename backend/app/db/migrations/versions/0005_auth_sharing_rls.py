"""Authentication, sharing, audit log and Postgres row-level security.

- users.is_active; groups and memberships; document_acl grants; refresh tokens; audit log.
- PostgreSQL only: the `rag_app` login role the application connects as. It is not a superuser
  and has no BYPASSRLS, so row-level security applies to it (superusers always bypass RLS,
  which is why the app must not use the owner role).
- Row-level security on documents, document_pages and chunks. The policy reads the
  transaction-local settings rag.scope / rag.tenant_id / rag.user_id / rag.group_ids that the
  application sets per session (app.db.access.RlsContext). Without them, no rows are visible.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-29
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
APP_ROLE = "rag_app"
PROTECTED = ("documents", "document_pages", "chunks")


def _created_at() -> sa.Column[Any]:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False)
        )

    op.create_table(
        "groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_groups_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_groups"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_groups_tenant_id_name"),
    )
    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_user_groups_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["groups.id"], name="fk_user_groups_group_id_groups", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "group_id", name="pk_user_groups"),
    )
    op.create_index("ix_user_groups_group_id", "user_groups", ["group_id"])

    op.create_table(
        "document_acl",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("principal_type", sa.String(10), nullable=False),
        sa.Column("principal_id", sa.Uuid(), nullable=False),
        sa.Column("permission", sa.String(10), nullable=False),
        sa.Column("granted_by", sa.Uuid(), nullable=True),
        _created_at(),
        sa.CheckConstraint(
            "principal_type IN ('user', 'group')", name="ck_document_acl_principal_type_valid"
        ),
        sa.CheckConstraint(
            "permission IN ('read', 'write')", name="ck_document_acl_permission_valid"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_acl_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["granted_by"],
            ["users.id"],
            name="fk_document_acl_granted_by_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_acl"),
        sa.UniqueConstraint(
            "document_id",
            "principal_type",
            "principal_id",
            name="uq_document_acl_document_id_principal_type_principal_id",
        ),
    )
    op.create_index("ix_document_acl_principal", "document_acl", ["principal_type", "principal_id"])

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_refresh_tokens_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_refresh_tokens"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_family_id", "refresh_tokens", ["family_id"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("target_type", sa.String(50), nullable=True),
        sa.Column("target_id", sa.String(100), nullable=True),
        sa.Column("ip", sa.String(64), nullable=True),
        sa.Column("details", JSONType, nullable=False),
        _created_at(),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_audit_log_tenant_id_tenants", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_audit_log_user_id_users", ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_tenant_id_created_at", "audit_log", ["tenant_id", "created_at"])

    if op.get_bind().dialect.name == "postgresql":
        _create_app_role()
        _enable_row_level_security()


def _create_app_role() -> None:
    password = get_settings().app_db_password.get_secret_value().replace("'", "''")
    op.execute(
        f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
            CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{password}'
              NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
          END IF;
        END $$;
        """
    )
    op.execute(
        f"DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO {APP_ROLE}',"
        " current_database()); END $$;"
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    op.execute(f"REVOKE INSERT, UPDATE, DELETE ON alembic_version FROM {APP_ROLE}")
    # Tables added by later migrations (run by the owner) are granted automatically.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public"
        f" GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}"
    )


def _enable_row_level_security() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION rag_setting(name text) RETURNS text
        LANGUAGE sql STABLE AS $$ SELECT nullif(current_setting(name, true), '') $$
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION rag_can_read(
            doc_tenant uuid, doc_id uuid, doc_owner uuid, doc_visibility text
        ) RETURNS boolean
        LANGUAGE sql STABLE AS $$
          SELECT CASE
            WHEN rag_setting('rag.scope') = 'system' THEN true
            WHEN rag_setting('rag.tenant_id') IS NULL
                 OR doc_tenant <> rag_setting('rag.tenant_id')::uuid THEN false
            WHEN rag_setting('rag.scope') = 'tenant' THEN true
            WHEN rag_setting('rag.scope') = 'user' THEN coalesce(
                 doc_owner = rag_setting('rag.user_id')::uuid
                 OR doc_visibility = 'tenant'
                 OR EXISTS (
                   SELECT 1 FROM document_acl a
                   WHERE a.document_id = doc_id
                     AND ((a.principal_type = 'user'
                           AND a.principal_id = rag_setting('rag.user_id')::uuid)
                       OR (a.principal_type = 'group'
                           AND a.principal_id = ANY (
                             coalesce(rag_setting('rag.group_ids'), '{}')::uuid[])))
                 ), false)
            ELSE false
          END
        $$
        """
    )
    op.execute(
        """
        CREATE POLICY documents_access ON documents
          USING (rag_can_read(tenant_id, id, owner_id, visibility))
          WITH CHECK (rag_setting('rag.scope') = 'system'
                      OR tenant_id = rag_setting('rag.tenant_id')::uuid)
        """
    )
    for table in ("document_pages", "chunks"):
        # A page or chunk is visible exactly when its document is (that subquery is itself
        # filtered by the documents policy).
        # Table names come from a fixed tuple, never from input.
        op.execute(
            f"""
            CREATE POLICY {table}_access ON {table}
              USING (EXISTS (SELECT 1 FROM documents d WHERE d.id = {table}.document_id))
              WITH CHECK (EXISTS (SELECT 1 FROM documents d WHERE d.id = {table}.document_id))
            """  # noqa: S608
        )
    for table in PROTECTED:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for table in PROTECTED:
            op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
            op.execute(f"DROP POLICY IF EXISTS {table}_access ON {table}")
        op.execute("DROP FUNCTION IF EXISTS rag_can_read(uuid, uuid, uuid, text)")
        op.execute("DROP FUNCTION IF EXISTS rag_setting(text)")
        # The rag_app role is cluster-wide and may serve other databases; it is left in place.

    op.drop_index("ix_audit_log_tenant_id_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_refresh_tokens_family_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_document_acl_principal", table_name="document_acl")
    op.drop_table("document_acl")
    op.drop_index("ix_user_groups_group_id", table_name="user_groups")
    op.drop_table("user_groups")
    op.drop_table("groups")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("is_active")
