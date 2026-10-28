"""Chat on top of real retrieval (PostgreSQL + pgvector); only the LLM is a fake."""

import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, Role, User
from tests.conftest import FakeAnswerGenerator, FakeEmbeddingProvider, Identity, requires_postgres
from tests.integration.test_chat_api import ask, new_conversation
from tests.postgres.test_retrieval import LEAVE, SECURITY, ConceptEmbedder, upload

pytestmark = requires_postgres


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return ConceptEmbedder()


async def test_answer_is_grounded_in_retrieved_chunks(
    client: httpx.AsyncClient,
    tmp_path: Path,
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    leave_id = await upload(client, tmp_path, "leave.md", LEAVE)
    await upload(client, tmp_path, "security.md", SECURITY)

    conversation_id = await new_conversation(client)
    events = await ask(client, conversation_id, "PTO allowance")  # a paraphrase: vector search

    sources = events[0][1]["sources"]
    assert sources[0]["filename"] == "leave.md"
    assert sources[0]["heading_path"][:1] == ["Leave Policy"]

    # The model received the same passages, with paragraph-level segments for citations.
    question, blocks = answer_generator.calls[0]
    assert question == "PTO allowance"
    assert str(blocks[0].document_id) == leave_id
    assert all(b.segments for b in blocks)

    citation = next(payload for name, payload in events if name == "citation")
    with sync_sessions() as s:
        chunks = s.scalars(
            select(Chunk).where(Chunk.id.in_([uuid.UUID(c) for c in citation["chunk_ids"]]))
        ).all()
    assert chunks and {str(c.document_id) for c in chunks} == {leave_id}
    assert citation["page_start"] == min(c.page_start for c in chunks)
    assert events[-1][0] == "done"


async def test_other_users_get_not_found_and_no_llm_call(
    client: httpx.AsyncClient,
    tmp_path: Path,
    answer_generator: FakeAnswerGenerator,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    await upload(client, tmp_path, "security.md", SECURITY)  # private to alice
    with sync_sessions() as s, s.begin():
        bob = User(tenant_id=tenant_and_user[0], email="bob@acme.test", role=Role.EDITOR)
        s.add(bob)
        s.flush()
        identity.user_id = bob.id

    conversation_id = await new_conversation(client)
    events = await ask(client, conversation_id, "What does SEC-4471 mean?")
    assert events[0][1]["not_found"] is True
    assert events[0][1]["sources"] == []
    assert answer_generator.calls == []
