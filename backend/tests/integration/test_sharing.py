"""Sharing, roles, admin and IDOR checks at the API level (SQLite: application filters only;
the same rules backed by row-level security are tested on Postgres in tests/postgres)."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AuditLog, Group, Role, Tenant, User, UserGroup
from tests.conftest import Identity

DOC_BYTES = b"Quarterly budget: 1.2M EUR approved for the Northwind expansion."


@dataclass
class Cast:
    alice: uuid.UUID  # editor, owns the documents
    bob: uuid.UUID  # viewer, in the Finance group
    carol: uuid.UUID  # viewer
    dave: uuid.UUID  # viewer, nothing shared with him
    admin: uuid.UUID
    eve: uuid.UUID  # admin of another organisation
    finance: uuid.UUID
    other_group: uuid.UUID  # belongs to the other organisation


@pytest.fixture
def cast(
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
    identity: Identity,
) -> Cast:
    tenant, alice = tenant_and_user
    with sync_sessions() as s, s.begin():

        def user(email: str, role: Role, tenant_id: uuid.UUID = tenant) -> uuid.UUID:
            u = User(tenant_id=tenant_id, email=email, role=role)
            s.add(u)
            s.flush()
            return u.id

        other = Tenant(name="globex")
        s.add(other)
        s.flush()
        finance, other_group = (
            Group(tenant_id=tenant, name="Finance"),
            Group(tenant_id=other.id, name="X"),
        )
        s.add_all([finance, other_group])
        s.flush()
        people = Cast(
            alice=alice,
            bob=user("bob@acme.test", Role.VIEWER),
            carol=user("carol@acme.test", Role.VIEWER),
            dave=user("dave@acme.test", Role.VIEWER),
            admin=user("admin@acme.test", Role.ADMIN),
            eve=user("eve@globex.test", Role.ADMIN, other.id),
            finance=finance.id,
            other_group=other_group.id,
        )
        s.add(UserGroup(user_id=people.bob, group_id=finance.id))
    identity.user_id = alice
    return people


async def upload(client: httpx.AsyncClient, tmp_path: Path, name: str = "budget.txt") -> str:
    response = await client.post("/documents", files={"file": (name, DOC_BYTES, "text/plain")})
    assert response.status_code == 202, response.text
    doc_id: str = response.json()["document"]["id"]
    return doc_id


async def status_as(
    client: httpx.AsyncClient,
    identity: Identity,
    user: uuid.UUID,
    method: str,
    url: str,
    **kw: object,
) -> int:
    identity.user_id = user
    response = await client.request(method, url, **kw)  # type: ignore[arg-type]
    return response.status_code


async def test_sharing_with_a_user_read_then_write_then_revoked(
    client: httpx.AsyncClient, tmp_path: Path, identity: Identity, cast: Cast
) -> None:
    doc = await upload(client, tmp_path)
    assert await status_as(client, identity, cast.carol, "GET", f"/documents/{doc}") == 404

    identity.user_id = cast.alice
    access = (
        await client.post(
            f"/documents/{doc}/access",
            json={"principal_type": "user", "principal_id": str(cast.carol)},
        )
    ).json()
    (entry,) = access["entries"]
    assert (entry["name"], entry["permission"]) == ("carol@acme.test", "read")

    assert await status_as(client, identity, cast.carol, "GET", f"/documents/{doc}") == 200
    assert await status_as(client, identity, cast.carol, "GET", f"/documents/{doc}/file") == 200
    # Read access does not allow changing the document.
    assert (
        await status_as(client, identity, cast.carol, "POST", f"/documents/{doc}/reprocess") == 403
    )
    assert await status_as(client, identity, cast.carol, "DELETE", f"/documents/{doc}") == 403
    assert (
        await status_as(
            client,
            identity,
            cast.carol,
            "PATCH",
            f"/documents/{doc}",
            json={"visibility": "tenant"},
        )
        == 403
    )

    identity.user_id = cast.alice
    await client.post(
        f"/documents/{doc}/access",
        json={"principal_type": "user", "principal_id": str(cast.carol), "permission": "write"},
    )
    assert (
        await status_as(client, identity, cast.carol, "POST", f"/documents/{doc}/reprocess") == 202
    )
    # Write access still does not include deleting or re-sharing.
    assert await status_as(client, identity, cast.carol, "DELETE", f"/documents/{doc}") == 403
    assert (
        await status_as(
            client,
            identity,
            cast.carol,
            "POST",
            f"/documents/{doc}/access",
            json={"principal_type": "user", "principal_id": str(cast.dave)},
        )
        == 403
    )

    identity.user_id = cast.alice
    await client.delete(f"/documents/{doc}/access/{entry['id']}")
    assert await status_as(client, identity, cast.carol, "GET", f"/documents/{doc}") == 404


async def test_sharing_with_a_group_follows_membership(
    client: httpx.AsyncClient, tmp_path: Path, identity: Identity, cast: Cast
) -> None:
    doc = await upload(client, tmp_path)
    await client.post(
        f"/documents/{doc}/access",
        json={"principal_type": "group", "principal_id": str(cast.finance)},
    )
    assert await status_as(client, identity, cast.bob, "GET", f"/documents/{doc}") == 200
    assert await status_as(client, identity, cast.dave, "GET", f"/documents/{doc}") == 404

    identity.user_id = cast.admin
    await client.delete(f"/admin/groups/{cast.finance}/members/{cast.bob}")
    assert await status_as(client, identity, cast.bob, "GET", f"/documents/{doc}") == 404


async def test_tenant_visibility_never_crosses_organisations(
    client: httpx.AsyncClient, tmp_path: Path, identity: Identity, cast: Cast
) -> None:
    doc = await upload(client, tmp_path)
    await client.patch(f"/documents/{doc}", json={"visibility": "tenant"})
    assert await status_as(client, identity, cast.dave, "GET", f"/documents/{doc}") == 200
    assert await status_as(client, identity, cast.eve, "GET", f"/documents/{doc}") == 404

    identity.user_id = cast.alice
    await client.patch(f"/documents/{doc}", json={"visibility": "private"})
    assert await status_as(client, identity, cast.dave, "GET", f"/documents/{doc}") == 404


async def test_cannot_share_with_people_or_groups_of_another_organisation(
    client: httpx.AsyncClient, tmp_path: Path, cast: Cast
) -> None:
    doc = await upload(client, tmp_path)
    for kind, target in (("user", cast.eve), ("group", cast.other_group), ("user", uuid.uuid4())):
        response = await client.post(
            f"/documents/{doc}/access", json={"principal_type": kind, "principal_id": str(target)}
        )
        assert response.status_code == 404


async def test_every_document_route_answers_404_to_outsiders(
    client: httpx.AsyncClient, tmp_path: Path, identity: Identity, cast: Cast
) -> None:
    doc = await upload(client, tmp_path)
    entry = (
        await client.post(
            f"/documents/{doc}/access",
            json={"principal_type": "user", "principal_id": str(cast.carol)},
        )
    ).json()["entries"][0]["id"]
    routes = [
        ("GET", f"/documents/{doc}", {}),
        ("GET", f"/documents/{doc}/file", {}),
        ("GET", f"/documents/{doc}/pages", {}),
        ("GET", f"/documents/{doc}/chunks", {}),
        ("GET", f"/documents/{doc}/access", {}),
        (
            "POST",
            f"/documents/{doc}/access",
            {"json": {"principal_type": "user", "principal_id": str(cast.dave)}},
        ),
        ("DELETE", f"/documents/{doc}/access/{entry}", {}),
        ("PATCH", f"/documents/{doc}", {"json": {"visibility": "tenant"}}),
        ("POST", f"/documents/{doc}/reprocess", {}),
        ("POST", f"/documents/{doc}/near-duplicate", {"json": {"action": "keep_both"}}),
        ("DELETE", f"/documents/{doc}", {}),
    ]
    for outsider in (cast.dave, cast.eve, cast.admin):  # admins get no extra read access
        for method, url, kw in routes:
            assert await status_as(client, identity, outsider, method, url, **kw) == 404, (
                outsider,
                method,
                url,
            )
    identity.user_id = cast.alice
    assert (await client.get(f"/documents/{doc}")).status_code == 200  # still intact


async def test_download_returns_the_original_file(
    client: httpx.AsyncClient, tmp_path: Path, cast: Cast
) -> None:
    doc = await upload(client, tmp_path, "budget 2026.txt")
    response = await client.get(f"/documents/{doc}/file")
    assert response.status_code == 200
    assert response.content == DOC_BYTES
    assert unquote(response.headers["content-disposition"]).endswith("budget 2026.txt")


async def test_viewers_cannot_upload(
    client: httpx.AsyncClient, identity: Identity, cast: Cast
) -> None:
    identity.user_id = cast.bob
    response = await client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 403


async def test_admin_manages_users_and_groups_of_their_own_organisation_only(
    client: httpx.AsyncClient, identity: Identity, cast: Cast, sync_sessions: sessionmaker[Session]
) -> None:
    identity.user_id = cast.bob
    assert (await client.get("/admin/users")).status_code == 403

    identity.user_id = cast.admin
    created = await client.post(
        "/admin/users",
        json={"email": "new@example.com", "password": "a long enough password", "role": "editor"},
    )
    assert created.status_code == 201
    again = await client.post(
        "/admin/users",
        json={"email": "new@example.com", "password": "a long enough password"},
    )
    assert again.status_code == 409
    emails = {u["email"] for u in (await client.get("/admin/users")).json()}
    assert "new@example.com" in emails and "eve@globex.test" not in emails
    bob = next(
        u for u in (await client.get("/admin/users")).json() if u["email"] == "bob@acme.test"
    )
    assert bob["groups"] == [str(cast.finance)]

    assert (
        await client.patch(f"/admin/users/{cast.admin}", json={"role": "viewer"})
    ).status_code == 409
    assert (
        await client.patch(f"/admin/users/{cast.eve}", json={"role": "viewer"})
    ).status_code == 404
    group = (await client.post("/admin/groups", json={"name": "Legal"})).json()
    assert (await client.put(f"/admin/groups/{group['id']}/members/{cast.eve}")).status_code == 404
    members = (await client.put(f"/admin/groups/{group['id']}/members/{cast.dave}")).json()
    assert members["member_ids"] == [str(cast.dave)]

    identity.user_id = cast.eve  # the other organisation's admin
    assert (await client.put(f"/admin/groups/{cast.finance}/members/{cast.eve}")).status_code == 404
    assert (await client.delete(f"/admin/groups/{cast.finance}")).status_code == 404
    assert (await client.get("/admin/audit")).json() == []

    identity.user_id = cast.admin
    actions = {a["action"] for a in (await client.get("/admin/audit")).json()}
    assert {"user.create", "group.create", "group.add_member"} <= actions
    with sync_sessions() as s:
        assert s.scalar(select(UserGroup).where(UserGroup.user_id == cast.dave)) is not None
        assert {a.tenant_id for a in s.scalars(select(AuditLog))} == {
            s.scalar(select(User.tenant_id).where(User.id == cast.admin))
        }
