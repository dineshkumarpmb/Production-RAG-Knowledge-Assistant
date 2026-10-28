"""Administration within the admin's own organisation: users, groups, audit log, reports.

Admins manage people and groups; they do not get extra read access to document content.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AdminPrincipal, DbSession
from app.api.routes.auth import Password, revoke_user_tokens
from app.api.schemas import IngestionReport
from app.core.audit import audit
from app.core.errors import AppError, NotFoundError
from app.core.security import hash_password
from app.db.access import Principal, RlsContext, set_rls_async
from app.db.models import (
    AuditLog,
    Chunk,
    Document,
    DocumentPage,
    Group,
    Role,
    User,
    UserGroup,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class UserCreate(Password):
    email: EmailStr
    role: Role = Role.VIEWER


class UserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime
    groups: list[uuid.UUID] = []


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class GroupOut(BaseModel):
    id: uuid.UUID
    name: str
    member_ids: list[uuid.UUID]


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    target_type: str | None
    target_id: str | None
    ip: str | None
    details: dict[str, Any]
    created_at: datetime


async def _tenant_user(db: DbSession, admin: Principal, user_id: uuid.UUID) -> User:
    user = await db.get(User, user_id)
    if user is None or user.tenant_id != admin.tenant_id:
        raise NotFoundError("User")
    return user


async def _tenant_group(db: DbSession, admin: Principal, group_id: uuid.UUID) -> Group:
    group = await db.get(Group, group_id)
    if group is None or group.tenant_id != admin.tenant_id:
        raise NotFoundError("Group")
    return group


# --- users ---------------------------------------------------------------------------------


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate, request: Request, db: DbSession, admin: AdminPrincipal
) -> UserOut:
    user = User(
        tenant_id=admin.tenant_id,
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise AppError("already_exists", "A user with that email already exists.", 409) from exc
    audit(
        db,
        "user.create",
        tenant_id=admin.tenant_id,
        user_id=admin.user_id,
        target_type="user",
        target_id=user.id,
        request=request,
        role=body.role,
    )
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.get("/users")
async def list_users(db: DbSession, admin: AdminPrincipal) -> list[UserOut]:
    users = list(
        await db.scalars(select(User).where(User.tenant_id == admin.tenant_id).order_by(User.email))
    )
    memberships: dict[uuid.UUID, list[uuid.UUID]] = {}
    for user_id, group_id in await db.execute(
        select(UserGroup.user_id, UserGroup.group_id).where(
            UserGroup.user_id.in_([u.id for u in users])
        )
    ):
        memberships.setdefault(user_id, []).append(group_id)
    return [
        UserOut.model_validate(u).model_copy(update={"groups": memberships.get(u.id, [])})
        for u in users
    ]


@router.patch("/users/{user_id}")
async def update_user(
    user_id: uuid.UUID, body: UserUpdate, request: Request, db: DbSession, admin: AdminPrincipal
) -> UserOut:
    """Change a user's role or deactivate them (which also signs them out everywhere)."""
    user = await _tenant_user(db, admin, user_id)
    if user.id == admin.user_id:
        # Prevents an organisation from locking itself out of administration.
        raise AppError("cannot_modify_self", "You cannot change your own role or status.", 409)
    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
        if not body.is_active:
            await revoke_user_tokens(db, user.id)
    audit(
        db,
        "user.update",
        tenant_id=admin.tenant_id,
        user_id=admin.user_id,
        target_type="user",
        target_id=user.id,
        request=request,
        role=body.role,
        is_active=body.is_active,
    )
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


# --- groups --------------------------------------------------------------------------------


async def _group_out(db: DbSession, group: Group) -> GroupOut:
    members = await db.scalars(select(UserGroup.user_id).where(UserGroup.group_id == group.id))
    return GroupOut(id=group.id, name=group.name, member_ids=list(members))


@router.post("/groups", status_code=status.HTTP_201_CREATED)
async def create_group(
    body: GroupCreate, request: Request, db: DbSession, admin: AdminPrincipal
) -> GroupOut:
    group = Group(tenant_id=admin.tenant_id, name=body.name.strip())
    db.add(group)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise AppError("already_exists", "A group with that name already exists.", 409) from exc
    audit(
        db,
        "group.create",
        tenant_id=admin.tenant_id,
        user_id=admin.user_id,
        target_type="group",
        target_id=group.id,
        request=request,
        name=group.name,
    )
    await db.commit()
    return GroupOut(id=group.id, name=group.name, member_ids=[])


@router.get("/groups")
async def list_groups(db: DbSession, admin: AdminPrincipal) -> list[GroupOut]:
    groups = await db.scalars(
        select(Group).where(Group.tenant_id == admin.tenant_id).order_by(Group.name)
    )
    return [await _group_out(db, g) for g in groups]


@router.delete("/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_group(
    group_id: uuid.UUID, request: Request, db: DbSession, admin: AdminPrincipal
) -> Response:
    group = await _tenant_group(db, admin, group_id)
    await db.execute(delete(UserGroup).where(UserGroup.group_id == group.id))
    await db.delete(group)
    audit(
        db,
        "group.delete",
        tenant_id=admin.tenant_id,
        user_id=admin.user_id,
        target_type="group",
        target_id=group_id,
        request=request,
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/groups/{group_id}/members/{user_id}")
async def add_member(
    group_id: uuid.UUID, user_id: uuid.UUID, request: Request, db: DbSession, admin: AdminPrincipal
) -> GroupOut:
    group = await _tenant_group(db, admin, group_id)
    user = await _tenant_user(db, admin, user_id)
    if await db.get(UserGroup, (user.id, group.id)) is None:
        db.add(UserGroup(user_id=user.id, group_id=group.id))
        audit(
            db,
            "group.add_member",
            tenant_id=admin.tenant_id,
            user_id=admin.user_id,
            target_type="group",
            target_id=group.id,
            request=request,
            member=str(user.id),
        )
        await db.commit()
    return await _group_out(db, group)


@router.delete("/groups/{group_id}/members/{user_id}")
async def remove_member(
    group_id: uuid.UUID, user_id: uuid.UUID, request: Request, db: DbSession, admin: AdminPrincipal
) -> GroupOut:
    group = await _tenant_group(db, admin, group_id)
    membership = await db.get(UserGroup, (user_id, group.id))
    if membership is not None:
        await db.delete(membership)
        audit(
            db,
            "group.remove_member",
            tenant_id=admin.tenant_id,
            user_id=admin.user_id,
            target_type="group",
            target_id=group.id,
            request=request,
            member=str(user_id),
        )
        await db.commit()
    return await _group_out(db, group)


# --- audit log and reports -----------------------------------------------------------------


@router.get("/audit")
async def audit_log(
    db: DbSession,
    admin: AdminPrincipal,
    action: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditOut]:
    query = select(AuditLog).where(AuditLog.tenant_id == admin.tenant_id)
    if action:
        query = query.where(AuditLog.action == action)
    rows = await db.scalars(
        query.order_by(AuditLog.created_at.desc(), AuditLog.id).limit(limit).offset(offset)
    )
    return [AuditOut.model_validate(r) for r in rows]


@router.get("/ingestion-report")
async def ingestion_report(db: DbSession, admin: AdminPrincipal) -> IngestionReport:
    """Corpus-wide ingestion numbers for the admin's tenant: chunks, duplicates, embedding cost.
    Counts every document of the tenant (tenant-wide RLS scope); returns no content."""
    tenant = admin.tenant_id
    await set_rls_async(db, RlsContext.tenant_wide(tenant))
    by_status = {
        doc_status: count
        for doc_status, count in await db.execute(
            select(Document.status, func.count())
            .where(Document.tenant_id == tenant)
            .group_by(Document.status)
        )
    }
    pages_total = await db.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .join(Document, Document.id == DocumentPage.document_id)
        .where(Document.tenant_id == tenant)
    )
    chunks_total, distinct_texts, avg_tokens = (
        await db.execute(
            select(func.count(), func.count(distinct(Chunk.text_hash)), func.avg(Chunk.token_count))
            .select_from(Chunk)
            .where(Chunk.tenant_id == tenant)
        )
    ).one()
    by_model = {
        model or "none": count
        for model, count in await db.execute(
            select(Chunk.embedding_model, func.count())
            .where(Chunk.tenant_id == tenant)
            .group_by(Chunk.embedding_model)
        )
    }
    near_pending = await db.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.tenant_id == tenant, Document.near_duplicate_of_id.is_not(None))
    )

    # Per-document stats live in JSON metadata; summing in Python keeps this database-agnostic.
    metas = list(await db.scalars(select(Document.meta).where(Document.tenant_id == tenant)))
    embedding = [m.get("embedding", {}) for m in metas]
    return IngestionReport(
        documents_total=sum(by_status.values()),
        documents_by_status=by_status,
        pages_total=pages_total or 0,
        chunks_total=chunks_total,
        avg_chunk_tokens=round(float(avg_tokens), 1) if avg_tokens is not None else None,
        duplicate_chunks=chunks_total - distinct_texts,
        chunks_by_embedding_model=by_model,
        exact_duplicate_uploads=sum(m.get("duplicate_uploads", 0) for m in metas),
        near_duplicates_pending=near_pending or 0,
        embedding_tokens=sum(e.get("tokens", 0) for e in embedding),
        embedding_cost_usd=round(sum(e.get("cost_usd", 0.0) for e in embedding), 6),
        embedding_cache_hits=sum(e.get("cache_hits", 0) for e in embedding),
    )
