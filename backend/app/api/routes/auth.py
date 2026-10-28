"""Sign-up, sign-in and tokens.

- Access tokens: JWTs valid for ACCESS_TOKEN_MINUTES (default 15), sent as `Authorization: Bearer`.
- Refresh tokens: opaque, stored only as a SHA-256 hash, valid REFRESH_TOKEN_DAYS (default 14).
  Every refresh rotates the token. Presenting one that was already rotated means it was copied,
  so the whole token family is revoked and the user must sign in again.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession, LimiterDep, SettingsDep
from app.core.audit import audit, client_ip
from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import (
    MIN_PASSWORD_LENGTH,
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)
from app.db.models import Group, RefreshToken, Role, Tenant, User, UserGroup

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_BAD_LOGIN = "Incorrect email or password."
_BAD_REFRESH = "The refresh token is invalid or has expired. Please sign in again."


class Password(BaseModel):
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)


class RegisterRequest(Password):
    organisation: str = Field(min_length=2, max_length=200)
    email: EmailStr


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=200)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 (OAuth2 token type, not a secret)
    expires_in: int  # seconds until the access token expires


class Me(BaseModel):
    id: uuid.UUID
    email: str
    role: str
    tenant_id: uuid.UUID
    organisation: str
    groups: list[str]


async def issue_tokens(
    db: AsyncSession, settings: Settings, user: User, family_id: uuid.UUID | None = None
) -> TokenPair:
    token, token_hash = new_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            family_id=family_id or uuid.uuid4(),
            token_hash=token_hash,
            expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_days),
        )
    )
    access = create_access_token(
        settings, user_id=user.id, tenant_id=user.tenant_id, role=user.role
    )
    return TokenPair(
        access_token=access, refresh_token=token, expires_in=settings.access_token_minutes * 60
    )


async def revoke_family(db: AsyncSession, family_id: uuid.UUID) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


async def revoke_user_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest, request: Request, db: DbSession, settings: SettingsDep
) -> TokenPair:
    """Create an organisation and its first user, who becomes its administrator."""
    if not settings.allow_registration:
        raise AppError("registration_closed", "Self-service registration is disabled.", 403)
    tenant = Tenant(name=body.organisation.strip())
    db.add(tenant)
    try:
        await db.flush()
        user = User(
            tenant_id=tenant.id,
            email=body.email.lower(),
            password_hash=hash_password(body.password),
            role=Role.ADMIN,
        )
        db.add(user)
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise AppError(
            "already_exists", "That organisation name or email is already registered.", 409
        ) from exc
    tokens = await issue_tokens(db, settings, user)
    audit(db, "auth.register", tenant_id=tenant.id, user_id=user.id, request=request)
    await db.commit()
    return tokens


@router.post("/login")
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    db: DbSession,
    settings: SettingsDep,
    limiter: LimiterDep,
) -> TokenPair:
    """OAuth2 password flow: `username` is the email address."""
    email = form.username.strip().lower()
    retry_after = await limiter.hit(
        f"login:{email}:{client_ip(request)}", settings.rate_limit_login_per_minute
    )
    if retry_after is not None:
        raise AppError(
            "rate_limited",
            "Too many sign-in attempts. Please wait a moment.",
            429,
            headers={"Retry-After": str(retry_after)},
        )

    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    ok = verify_password(form.password, user.password_hash if user else None)
    if user is None or not ok or not user.is_active:
        audit(
            db,
            "auth.login_failed",
            tenant_id=user.tenant_id if user else None,
            user_id=user.id if user else None,
            request=request,
            email=email,
        )
        await db.commit()
        raise AppError("invalid_credentials", _BAD_LOGIN, 401)

    tokens = await issue_tokens(db, settings, user)
    audit(db, "auth.login", tenant_id=user.tenant_id, user_id=user.id, request=request)
    await db.commit()
    return tokens


@router.post("/refresh")
async def refresh(
    body: RefreshRequest, request: Request, db: DbSession, settings: SettingsDep
) -> TokenPair:
    stored = (
        await db.execute(
            select(RefreshToken)
            .where(RefreshToken.token_hash == hash_refresh_token(body.refresh_token))
            .with_for_update()
        )
    ).scalar_one_or_none()
    if stored is None:
        raise AppError("invalid_refresh_token", _BAD_REFRESH, 401)
    user = await db.get(User, stored.user_id)

    if stored.revoked_at is not None:
        # This token was already rotated: someone is replaying a copy. End the whole session.
        await revoke_family(db, stored.family_id)
        audit(
            db,
            "auth.refresh_reuse",
            tenant_id=user.tenant_id if user else None,
            user_id=stored.user_id,
            request=request,
        )
        await db.commit()
        log.warning("auth.refresh_reuse", user_id=str(stored.user_id))
        raise AppError("invalid_refresh_token", _BAD_REFRESH, 401)

    expires_at = (
        stored.expires_at if stored.expires_at.tzinfo else stored.expires_at.replace(tzinfo=UTC)
    )
    if expires_at <= datetime.now(UTC) or user is None or not user.is_active:
        raise AppError("invalid_refresh_token", _BAD_REFRESH, 401)

    stored.revoked_at = datetime.now(UTC)
    tokens = await issue_tokens(db, settings, user, family_id=stored.family_id)
    await db.commit()
    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, db: DbSession) -> Response:
    """Revoke the refresh token's whole family (this browser's session)."""
    stored = (
        await db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == hash_refresh_token(body.refresh_token)
            )
        )
    ).scalar_one_or_none()
    if stored is not None:
        await revoke_family(db, stored.family_id)
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me")
async def me(user: CurrentUser, db: DbSession) -> Me:
    tenant = await db.get(Tenant, user.tenant_id)
    groups = await db.scalars(
        select(Group.name)
        .join(UserGroup, UserGroup.group_id == Group.id)
        .where(UserGroup.user_id == user.id)
        .order_by(Group.name)
    )
    return Me(
        id=user.id,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id,
        organisation=tenant.name if tenant else "",
        groups=list(groups),
    )
