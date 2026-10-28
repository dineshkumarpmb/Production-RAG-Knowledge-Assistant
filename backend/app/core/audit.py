"""Audit trail of security-relevant actions (sign-ins, uploads, sharing, deletions, queries)."""

import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog


def client_ip(request: Request | None) -> str | None:
    return request.client.host if request is not None and request.client else None


def audit(
    db: AsyncSession,
    action: str,
    *,
    tenant_id: uuid.UUID | None,
    user_id: uuid.UUID | None,
    target_type: str | None = None,
    target_id: object | None = None,
    request: Request | None = None,
    **details: Any,
) -> None:
    """Add an audit entry to the session; it is written with the caller's next commit."""
    db.add(
        AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            ip=client_ip(request),
            details={k: v for k, v in details.items() if v is not None},
        )
    )
