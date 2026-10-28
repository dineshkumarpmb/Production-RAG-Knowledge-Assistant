"""The single rule for which documents a user may read.

Every query that returns document content (the documents API, near-duplicate checks, keyword
and vector search) filters with `can_read`, so there is no code path that skips it. Phase 6
extends this rule with groups and per-document ACL entries.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, and_, or_

from app.db.models import Document, User, Visibility


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    tenant_id: uuid.UUID

    @classmethod
    def of(cls, user: User) -> "Principal":
        return cls(user_id=user.id, tenant_id=user.tenant_id)


def can_read(principal: Principal) -> ColumnElement[bool]:
    return and_(
        Document.tenant_id == principal.tenant_id,
        or_(Document.owner_id == principal.user_id, Document.visibility == Visibility.TENANT),
    )
