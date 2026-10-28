"""Authentication with real tokens (the identity override used elsewhere is removed here)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_current_user
from app.core.config import Settings, get_settings
from app.core.security import create_access_token, hash_password
from app.db.models import AuditLog, RefreshToken, Role, User

PASSWORD = "correct horse battery"


@pytest.fixture
def real_auth(app: FastAPI) -> FastAPI:
    app.dependency_overrides.pop(get_current_user)
    return app


async def register(
    client: httpx.AsyncClient, org: str = "Acme", email: str = "owner@example.com"
) -> dict[str, Any]:
    response = await client.post(
        "/auth/register", json={"organisation": org, "email": email, "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    tokens: dict[str, Any] = response.json()
    return tokens


async def login(client: httpx.AsyncClient, email: str, password: str = PASSWORD) -> httpx.Response:
    return await client.post("/auth/login", data={"username": email, "password": password})


def bearer(tokens: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def add_user(
    sessions: sessionmaker[Session], tenant_id: uuid.UUID, email: str, role: Role = Role.VIEWER
) -> uuid.UUID:
    with sessions() as s, s.begin():
        user = User(
            tenant_id=tenant_id, email=email, password_hash=hash_password(PASSWORD), role=role
        )
        s.add(user)
        s.flush()
        return user.id


async def test_register_creates_an_admin_who_can_use_the_api(
    client: httpx.AsyncClient, real_auth: FastAPI
) -> None:
    tokens = await register(client)
    assert tokens["token_type"] == "bearer" and tokens["expires_in"] == 15 * 60
    me = (await client.get("/auth/me", headers=bearer(tokens))).json()
    assert (me["email"], me["role"], me["organisation"]) == ("owner@example.com", "admin", "Acme")
    assert (await client.get("/documents", headers=bearer(tokens))).status_code == 200


async def test_registration_rules(
    client: httpx.AsyncClient, real_auth: FastAPI, settings: Settings
) -> None:
    await register(client)
    duplicate = await client.post(
        "/auth/register",
        json={"organisation": "Acme", "email": "other@example.com", "password": PASSWORD},
    )
    assert duplicate.status_code == 409
    weak = await client.post(
        "/auth/register",
        json={"organisation": "Beta", "email": "b@example.com", "password": "short"},
    )
    assert weak.status_code == 422

    real_auth.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"allow_registration": False}
    )
    closed = await client.post(
        "/auth/register",
        json={"organisation": "Gamma", "email": "g@example.com", "password": PASSWORD},
    )
    assert closed.status_code == 403


async def test_login_errors_do_not_reveal_which_accounts_exist(
    client: httpx.AsyncClient, real_auth: FastAPI, sync_sessions: sessionmaker[Session]
) -> None:
    await register(client)
    wrong_password = await login(client, "owner@example.com", "wrong password!!")
    no_such_user = await login(client, "nobody@example.com")
    assert wrong_password.status_code == no_such_user.status_code == 401
    assert wrong_password.json()["message"] == no_such_user.json()["message"]

    ok = await login(client, "OWNER@example.com")  # emails are case-insensitive
    assert ok.status_code == 200
    with sync_sessions() as s:
        actions = [a.action for a in s.scalars(select(AuditLog).order_by(AuditLog.created_at))]
    assert actions == ["auth.register", "auth.login_failed", "auth.login_failed", "auth.login"]


async def test_missing_bad_and_forged_tokens_are_rejected(
    client: httpx.AsyncClient, real_auth: FastAPI, settings: Settings
) -> None:
    tokens = await register(client)
    me = (await client.get("/auth/me", headers=bearer(tokens))).json()
    user_id, tenant_id = uuid.UUID(me["id"]), uuid.UUID(me["tenant_id"])

    missing = await client.get("/documents")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    garbage = await client.get("/documents", headers={"Authorization": "Bearer not-a-jwt"})
    assert garbage.status_code == 401

    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "tid": str(tenant_id), "role": "admin", "type": "access"}
    forged = jwt.encode({**claims, "exp": now + timedelta(minutes=5)}, "wrong-secret", "HS256")
    expired = jwt.encode(
        {**claims, "exp": now - timedelta(seconds=1)},
        settings.jwt_secret.get_secret_value(),
        "HS256",
    )
    unsigned = jwt.encode({**claims, "exp": now + timedelta(minutes=5)}, None, algorithm="none")
    not_access = jwt.encode(
        {**claims, "type": "refresh", "exp": now + timedelta(minutes=5)},
        settings.jwt_secret.get_secret_value(),
        "HS256",
    )
    for token in (forged, expired, unsigned, not_access):
        response = await client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401, token


async def test_refresh_rotates_and_reuse_revokes_the_family(
    client: httpx.AsyncClient, real_auth: FastAPI, sync_sessions: sessionmaker[Session]
) -> None:
    first = await register(client)
    second = (
        await client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    ).json()
    assert second["refresh_token"] != first["refresh_token"]
    assert (await client.get("/auth/me", headers=bearer(second))).status_code == 200

    # Replaying the rotated token (e.g. stolen) kills the whole session family...
    replay = await client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert replay.status_code == 401
    # ...including the legitimate latest token.
    latest = await client.post("/auth/refresh", json={"refresh_token": second["refresh_token"]})
    assert latest.status_code == 401
    with sync_sessions() as s:
        assert all(t.revoked_at is not None for t in s.scalars(select(RefreshToken)))
        assert s.scalar(select(AuditLog).where(AuditLog.action == "auth.refresh_reuse"))


async def test_logout_revokes_the_refresh_token(
    client: httpx.AsyncClient, real_auth: FastAPI
) -> None:
    tokens = await register(client)
    assert (
        await client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    ).status_code == 204
    again = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert again.status_code == 401


async def test_login_is_rate_limited(client: httpx.AsyncClient, real_auth: FastAPI) -> None:
    await register(client)
    statuses = [
        (await login(client, "owner@example.com", "wrong password!!")).status_code
        for _ in range(11)
    ]
    assert statuses[:10] == [401] * 10
    assert statuses[10] == 429
    blocked = await login(client, "owner@example.com")
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0


async def test_role_and_status_changes_apply_immediately(
    client: httpx.AsyncClient, real_auth: FastAPI, sync_sessions: sessionmaker[Session]
) -> None:
    admin = await register(client)
    tenant_id = uuid.UUID((await client.get("/auth/me", headers=bearer(admin))).json()["tenant_id"])
    editor_id = add_user(sync_sessions, tenant_id, "ed@example.com", Role.EDITOR)
    editor = (await login(client, "ed@example.com")).json()

    upload = {"file": ("a.txt", b"hello world", "text/plain")}
    assert (
        await client.post("/documents", files=upload, headers=bearer(editor))
    ).status_code == 202

    demote = await client.patch(
        f"/admin/users/{editor_id}", json={"role": "viewer"}, headers=bearer(admin)
    )
    assert demote.status_code == 200
    # The token still says "editor", but the database is the source of truth.
    upload = {"file": ("b.txt", b"another file", "text/plain")}
    assert (
        await client.post("/documents", files=upload, headers=bearer(editor))
    ).status_code == 403

    deactivate = await client.patch(
        f"/admin/users/{editor_id}", json={"is_active": False}, headers=bearer(admin)
    )
    assert deactivate.status_code == 200
    assert (await client.get("/documents", headers=bearer(editor))).status_code == 401
    refresh = await client.post("/auth/refresh", json={"refresh_token": editor["refresh_token"]})
    assert refresh.status_code == 401
    assert (await login(client, "ed@example.com")).status_code == 401


async def test_tokens_from_another_tenant_claim_are_rejected(
    client: httpx.AsyncClient, real_auth: FastAPI, settings: Settings
) -> None:
    tokens = await register(client)
    me = (await client.get("/auth/me", headers=bearer(tokens))).json()
    wrong_tenant = create_access_token(
        settings, user_id=uuid.UUID(me["id"]), tenant_id=uuid.uuid4(), role="admin"
    )
    response = await client.get("/documents", headers={"Authorization": f"Bearer {wrong_tenant}"})
    assert response.status_code == 401


async def test_security_headers_and_cors(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    other = await client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    docs = await client.get("/docs")
    assert "content-security-policy" not in docs.headers  # Swagger UI loads its own scripts
