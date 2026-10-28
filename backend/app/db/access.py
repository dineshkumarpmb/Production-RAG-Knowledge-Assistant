"""Who may read or change which documents.

Two layers, so one mistake cannot leak a document:

1. `can_read` / `can_write`: SQL conditions every query on documents applies (documents API,
   near-duplicate checks, keyword and vector search).
2. Postgres row-level security (migration 0005) on `documents`, `document_pages` and `chunks`,
   driven by the session's `RlsContext`. A session without a context sees no rows at all.

A user can read a document of their own tenant when they own it, it is shared with the whole
tenant (`visibility = tenant`), or a `document_acl` entry grants it to them or to one of their
groups. Admins manage users and groups but get no extra read access to content.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import ColumnElement, and_, event, exists, or_, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, SessionTransaction

from app.db.models import Document, DocumentAcl, Permission, PrincipalType, Role, User, Visibility


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    group_ids: tuple[uuid.UUID, ...] = ()
    role: str = Role.VIEWER

    @classmethod
    def of(cls, user: User, group_ids: tuple[uuid.UUID, ...] = ()) -> "Principal":
        return cls(user_id=user.id, tenant_id=user.tenant_id, group_ids=group_ids, role=user.role)

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN


def _granted(principal: Principal, permissions: tuple[str, ...]) -> ColumnElement[bool]:
    grantee = and_(
        DocumentAcl.principal_type == PrincipalType.USER,
        DocumentAcl.principal_id == principal.user_id,
    )
    if principal.group_ids:
        grantee = or_(
            grantee,
            and_(
                DocumentAcl.principal_type == PrincipalType.GROUP,
                DocumentAcl.principal_id.in_(principal.group_ids),
            ),
        )
    return exists(
        select(DocumentAcl.id).where(
            DocumentAcl.document_id == Document.id,
            DocumentAcl.permission.in_(permissions),
            grantee,
        )
    )


def can_read(principal: Principal) -> ColumnElement[bool]:
    return and_(
        Document.tenant_id == principal.tenant_id,
        or_(
            Document.owner_id == principal.user_id,
            Document.visibility == Visibility.TENANT,
            _granted(principal, (Permission.READ, Permission.WRITE)),
        ),
    )


def can_write(principal: Principal) -> ColumnElement[bool]:
    return and_(
        Document.tenant_id == principal.tenant_id,
        or_(Document.owner_id == principal.user_id, _granted(principal, (Permission.WRITE,))),
    )


# --- row-level security context ----------------------------------------------------------

RlsScope = Literal["user", "tenant", "system"]
_RLS_KEY = "rls_context"
_SET_RLS = text(
    "SELECT set_config('rag.scope', :scope, true), set_config('rag.tenant_id', :tenant, true),"
    " set_config('rag.user_id', :user, true), set_config('rag.group_ids', :groups, true)"
)


@dataclass(frozen=True)
class RlsContext:
    scope: RlsScope
    tenant_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    group_ids: tuple[uuid.UUID, ...] = ()

    @classmethod
    def for_principal(cls, principal: Principal) -> "RlsContext":
        return cls("user", principal.tenant_id, principal.user_id, principal.group_ids)

    @classmethod
    def tenant_wide(cls, tenant_id: uuid.UUID) -> "RlsContext":
        """Every document of one tenant: admin reports and duplicate detection."""
        return cls("tenant", tenant_id)

    @classmethod
    def system(cls) -> "RlsContext":
        """No restriction: the ingestion worker and maintenance tasks."""
        return cls("system")

    def params(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "tenant": str(self.tenant_id or ""),
            "user": str(self.user_id or ""),
            "groups": "{" + ",".join(str(g) for g in self.group_ids) + "}",
        }


def _apply(connection: Connection, context: RlsContext | None) -> None:
    if context is not None and connection.dialect.name == "postgresql":
        connection.execute(_SET_RLS, context.params())


@event.listens_for(Session, "after_begin")
def _set_rls_on_begin(session: Session, _: SessionTransaction, connection: Connection) -> None:
    # The settings are transaction-local (SET LOCAL), so they never leak to another request
    # through the connection pool, and must be re-applied in every transaction.
    _apply(connection, session.info.get(_RLS_KEY))


def set_rls(session: Session, context: RlsContext) -> None:
    session.info[_RLS_KEY] = context
    if session.in_transaction():
        _apply(session.connection(), context)


async def set_rls_async(session: AsyncSession, context: RlsContext) -> None:
    session.info[_RLS_KEY] = context
    if session.in_transaction() and session.sync_session.get_bind().dialect.name == "postgresql":
        await session.execute(_SET_RLS, context.params())


def rls_session_info(context: RlsContext) -> dict[str, Any]:
    """`info` for a sessionmaker whose sessions should all use one context."""
    return {_RLS_KEY: context}
