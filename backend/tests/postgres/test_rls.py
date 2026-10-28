"""Postgres row-level security on its own: raw queries with no application filter at all, as
the unprivileged `rag_app` role. If a code path ever forgets `can_read`, this is what stops a
leak."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.access import RlsContext, set_rls
from app.db.models import (
    Chunk,
    Document,
    DocumentAcl,
    DocumentPage,
    Group,
    Role,
    Tenant,
    User,
    UserGroup,
)
from app.db.session import sync_url
from tests.conftest import requires_postgres

pytestmark = requires_postgres


@dataclass
class World:
    acme: uuid.UUID
    globex: uuid.UUID
    alice: uuid.UUID
    bob: uuid.UUID
    dave: uuid.UUID
    eve: uuid.UUID
    finance: uuid.UUID


@pytest.fixture
def world(sync_sessions: sessionmaker[Session]) -> World:
    """Documents named after who can read them, each with one page and one chunk."""
    with sync_sessions() as s, s.begin():
        acme, globex = Tenant(name="acme"), Tenant(name="globex")
        s.add_all([acme, globex])
        s.flush()
        users = {
            name: User(tenant_id=tenant, email=f"{name}@x.test", role=Role.VIEWER)
            for name, tenant in [
                ("alice", acme.id),
                ("bob", acme.id),
                ("dave", acme.id),
                ("eve", globex.id),
            ]
        }
        s.add_all(users.values())
        finance = Group(tenant_id=acme.id, name="Finance")
        s.add(finance)
        s.flush()
        s.add(UserGroup(user_id=users["bob"].id, group_id=finance.id))

        def document(
            name: str, tenant: uuid.UUID, owner: uuid.UUID, visibility: str = "private"
        ) -> uuid.UUID:
            doc = Document(
                tenant_id=tenant,
                owner_id=owner,
                filename=f"{name}.md",
                file_type="md",
                mime_type="text/markdown",
                size_bytes=1,
                sha256=uuid.uuid4().hex * 2,
                storage_uri="local://none",
                status="ready",
                visibility=visibility,
                meta={},
            )
            s.add(doc)
            s.flush()
            s.add(DocumentPage(document_id=doc.id, page_number=1, text=name, char_count=len(name)))
            s.add(
                Chunk(
                    document_id=doc.id,
                    tenant_id=tenant,
                    chunk_index=0,
                    text=name,
                    text_hash=uuid.uuid4().hex * 2,
                    page_start=1,
                    page_end=1,
                    char_start=0,
                    char_end=len(name),
                    heading_path=[],
                    token_count=1,
                )
            )
            return doc.id

        a, b, d, e = (users[n].id for n in ("alice", "bob", "dave", "eve"))
        document("alice-private", acme.id, a)
        shared = document("finance-shared", acme.id, a)
        s.add(
            DocumentAcl(
                document_id=shared,
                principal_type="group",
                principal_id=finance.id,
                permission="read",
            )
        )
        direct = document("dave-shared", acme.id, a)
        s.add(
            DocumentAcl(
                document_id=direct, principal_type="user", principal_id=d, permission="read"
            )
        )
        document("acme-wide", acme.id, a, visibility="tenant")
        document("globex-wide", globex.id, e, visibility="tenant")
        return World(acme.id, globex.id, a, b, d, e, finance.id)


@pytest.fixture
def app_sessions(settings: Settings) -> Iterator[sessionmaker[Session]]:
    """Plain sessions as `rag_app` with no RLS context: each test sets one explicitly."""
    engine = create_engine(sync_url(settings.database_url))
    yield sessionmaker(engine)
    engine.dispose()


def visible(sessions: sessionmaker[Session], context: RlsContext | None) -> dict[str, set[str]]:
    with sessions() as s, s.begin():
        if context is not None:
            set_rls(s, context)
        # Deliberately no WHERE clause: only row-level security decides.
        return {
            "documents": {
                f.removesuffix(".md") for f in s.scalars(text("SELECT filename FROM documents"))
            },
            "pages": set(s.scalars(text("SELECT text FROM document_pages"))),
            "chunks": set(s.scalars(text("SELECT text FROM chunks"))),
        }


def user(world: World, name: str, groups: tuple[uuid.UUID, ...] = ()) -> RlsContext:
    user_id = getattr(world, name)
    tenant = world.globex if name == "eve" else world.acme
    return RlsContext("user", tenant, user_id, groups)


@pytest.mark.parametrize(
    ("who", "expected"),
    [
        ("alice", {"alice-private", "finance-shared", "dave-shared", "acme-wide"}),
        ("bob", {"finance-shared", "acme-wide"}),
        ("dave", {"dave-shared", "acme-wide"}),
        ("eve", {"globex-wide"}),
    ],
)
def test_each_user_sees_exactly_their_documents_pages_and_chunks(
    world: World, app_sessions: sessionmaker[Session], who: str, expected: set[str]
) -> None:
    groups = (world.finance,) if who == "bob" else ()
    seen = visible(app_sessions, user(world, who, groups))
    assert seen == {"documents": expected, "pages": expected, "chunks": expected}


def test_group_access_needs_the_group_in_the_context(
    world: World, app_sessions: sessionmaker[Session]
) -> None:
    assert visible(app_sessions, user(world, "bob"))["chunks"] == {"acme-wide"}


def test_no_context_means_no_rows(world: World, app_sessions: sessionmaker[Session]) -> None:
    assert visible(app_sessions, None) == {"documents": set(), "pages": set(), "chunks": set()}


def test_tenant_and_system_scopes(world: World, app_sessions: sessionmaker[Session]) -> None:
    acme = visible(app_sessions, RlsContext.tenant_wide(world.acme))["documents"]
    assert acme == {"alice-private", "finance-shared", "dave-shared", "acme-wide"}
    everything = visible(app_sessions, RlsContext.system())["documents"]
    assert everything == acme | {"globex-wide"}


def test_writes_cannot_target_another_tenant(
    world: World, app_sessions: sessionmaker[Session]
) -> None:
    with pytest.raises(DBAPIError, match="row-level security"), app_sessions() as s, s.begin():
        set_rls(s, user(world, "alice"))
        s.add(
            Document(
                tenant_id=world.globex,
                owner_id=world.alice,
                filename="sneaky.md",
                file_type="md",
                mime_type="text/markdown",
                size_bytes=1,
                sha256="f" * 64,
                storage_uri="local://none",
                meta={},
            )
        )
        s.flush()

    with app_sessions() as s, s.begin():
        set_rls(s, user(world, "dave"))
        # Updating rows you cannot see changes nothing.
        changed = s.execute(text("UPDATE documents SET visibility = 'tenant'")).rowcount
        assert changed == 2  # only dave-shared and acme-wide are visible to him
        s.rollback()


def test_settings_do_not_leak_between_transactions(
    world: World, app_sessions: sessionmaker[Session]
) -> None:
    """Scopes are transaction-local, so a pooled connection never inherits a previous user's."""
    engine = create_engine(sync_url(app_sessions.kw["bind"].url), pool_size=1, max_overflow=0)
    sessions = sessionmaker(engine)
    try:
        assert visible(sessions, user(world, "alice"))["documents"]
        assert visible(sessions, None)["documents"] == set()
    finally:
        engine.dispose()
