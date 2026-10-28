"""The permissions matrix: every user x every way of reaching document content, end to end
through the API on Postgres (application filters and row-level security both active).

Each cell must match exactly: an extra document is a leak, a missing one is broken sharing.
The rendered table is the one in the README.
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Group, Role, Tenant, User, UserGroup
from tests.conftest import Identity, requires_postgres
from tests.integration.test_chat_api import ask, new_conversation

pytestmark = requires_postgres

# One distinctive word per document, so every channel can report which documents it exposed.
DOCUMENTS = {
    "private": "zircon",  # alice only
    "group": "quartz",  # shared with the Finance group
    "user": "garnet",  # shared with carol
    "tenant": "basalt",  # visible to everyone at Acme
    "other-org": "obsidian",  # another organisation's tenant-wide document
}
EXPECTED = {
    "alice (owner)": {"private", "group", "user", "tenant"},
    "bob (Finance)": {"group", "tenant"},
    "carol (shared)": {"user", "tenant"},
    "dave (colleague)": {"tenant"},
    "admin (Acme)": {"tenant"},
    "eve (other org)": {"other-org"},
}
CHANNELS = [
    "list",
    "get",
    "pages",
    "chunks",
    "file",
    "search: keyword",
    "search: vector",
    "search: hybrid",
    "retrieve",
    "chat sources",
]
QUERY = " ".join(DOCUMENTS.values())


def render_matrix(seen: dict[str, dict[str, set[str]]]) -> str:
    """Markdown table: one row per user, one column per document, checked on every channel."""
    docs = list(DOCUMENTS)
    lines = [
        "| User | " + " | ".join(docs) + " |",
        "|---|" + "|".join(":---:" for _ in docs) + "|",
    ]
    for who, by_channel in seen.items():
        union = set().union(*by_channel.values())
        cells = ["✅" if d in union else "—" for d in docs]
        lines.append(f"| {who} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@pytest.fixture
def users(
    sync_sessions: sessionmaker[Session], tenant_and_user: tuple[uuid.UUID, uuid.UUID]
) -> dict[str, uuid.UUID]:
    acme, alice = tenant_and_user
    with sync_sessions() as s, s.begin():
        globex = Tenant(name="globex")
        s.add(globex)
        s.flush()
        made = {"alice (owner)": alice}
        for who, email, role, tenant in [
            ("bob (Finance)", "bob@acme.test", Role.VIEWER, acme),
            ("carol (shared)", "carol@acme.test", Role.VIEWER, acme),
            ("dave (colleague)", "dave@acme.test", Role.VIEWER, acme),
            ("admin (Acme)", "admin@acme.test", Role.ADMIN, acme),
            ("eve (other org)", "eve@globex.test", Role.ADMIN, globex.id),
        ]:
            u = User(tenant_id=tenant, email=email, role=role)
            s.add(u)
            s.flush()
            made[who] = u.id
        finance = Group(tenant_id=acme, name="Finance")
        s.add(finance)
        s.flush()
        s.add(UserGroup(user_id=made["bob (Finance)"], group_id=finance.id))
        made["finance"] = finance.id
    return made


async def setup_documents(
    client: httpx.AsyncClient, identity: Identity, users: dict[str, uuid.UUID]
) -> dict[str, str]:
    ids: dict[str, str] = {}
    for name, word in DOCUMENTS.items():
        identity.user_id = users["eve (other org)" if name == "other-org" else "alice (owner)"]
        body = f"# {name.title()} notes\n\nThis document mentions {word} several times: {word}.\n"
        response = await client.post(
            "/documents", files={"file": (f"{name}.md", body.encode(), "text/markdown")}
        )
        assert response.status_code == 202, response.text
        ids[name] = response.json()["document"]["id"]

    identity.user_id = users["alice (owner)"]
    share = [
        ("group", {"principal_type": "group", "principal_id": str(users["finance"])}),
        ("user", {"principal_type": "user", "principal_id": str(users["carol (shared)"])}),
    ]
    for name, body in share:
        assert (await client.post(f"/documents/{ids[name]}/access", json=body)).status_code == 200
    assert (
        await client.patch(f"/documents/{ids['tenant']}", json={"visibility": "tenant"})
    ).status_code == 200
    identity.user_id = users["eve (other org)"]
    assert (
        await client.patch(f"/documents/{ids['other-org']}", json={"visibility": "tenant"})
    ).status_code == 200
    return ids


async def observe(client: httpx.AsyncClient, ids: dict[str, str]) -> dict[str, set[str]]:
    """Which documents each channel exposes to the current user."""
    by_id = {v: k for k, v in ids.items()}
    by_word = {w: k for k, w in DOCUMENTS.items()}

    def from_words(texts: list[str]) -> set[str]:
        return {by_word[w] for t in texts for w in by_word if w in t}

    async def reachable(path: str) -> set[str]:
        return {
            name
            for name, doc in ids.items()
            if (await client.get(path.format(doc))).status_code == 200
        }

    async def search(mode: str) -> set[str]:
        hits = (
            await client.post("/search", json={"query": QUERY, "mode": mode, "top_k": 50})
        ).json()["hits"]
        return {by_id[h["document_id"]] for h in hits}

    async def retrieve() -> set[str]:
        blocks = (await client.post("/retrieve", json={"query": QUERY, "top_k": 50})).json()[
            "blocks"
        ]
        assert all(by_id[b["document_id"]] in from_words([b["text"]]) for b in blocks)
        return {by_id[b["document_id"]] for b in blocks}

    async def chat() -> set[str]:
        events = await ask(client, await new_conversation(client), QUERY)
        return {by_id[s["document_id"]] for s in events[0][1]["sources"]}

    async def listed() -> set[str]:
        return {by_id[d["id"]] for d in (await client.get("/documents")).json()["items"]}

    checks: dict[str, Callable[[], Awaitable[set[str]]]] = {
        "list": listed,
        "get": lambda: reachable("/documents/{}"),
        "pages": lambda: reachable("/documents/{}/pages"),
        "chunks": lambda: reachable("/documents/{}/chunks"),
        "file": lambda: reachable("/documents/{}/file"),
        "search: keyword": lambda: search("keyword"),
        "search: vector": lambda: search("vector"),
        "search: hybrid": lambda: search("hybrid"),
        "retrieve": retrieve,
        "chat sources": chat,
    }
    return {name: await check() for name, check in checks.items()}


async def test_permissions_matrix_has_zero_leaks(
    client: httpx.AsyncClient,
    identity: Identity,
    users: dict[str, uuid.UUID],
) -> None:
    ids = await setup_documents(client, identity, users)

    seen: dict[str, dict[str, set[str]]] = {}
    failures: list[Any] = []
    for who, expected in EXPECTED.items():
        identity.user_id = users[who]
        seen[who] = await observe(client, ids)
        for channel in CHANNELS:
            got = seen[who][channel]
            if got != expected:
                failures.append((who, channel, "leaked", got - expected, "missing", expected - got))

    assert failures == [], failures
    assert render_matrix(seen) == MATRIX  # the README shows this table


MATRIX = """\
| User | private | group | user | tenant | other-org |
|---|:---:|:---:|:---:|:---:|:---:|
| alice (owner) | ✅ | ✅ | ✅ | ✅ | — |
| bob (Finance) | — | ✅ | — | ✅ | — |
| carol (shared) | — | — | ✅ | ✅ | — |
| dave (colleague) | — | — | — | ✅ | — |
| admin (Acme) | — | — | — | ✅ | — |
| eve (other org) | — | — | — | — | ✅ |"""
