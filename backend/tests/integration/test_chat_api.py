"""The chat API's Server-Sent Events contract, with fake retrieval and a fake LLM (SQLite)."""

import json
import re
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes.search import get_retriever
from app.db.access import Principal
from app.db.models import QueryLog, Role, User
from app.generation.claude import LLMError
from app.generation.prompts import NOT_FOUND_ANSWER
from app.retrieval.context import ContextBlock, ContextSegment
from app.retrieval.service import RetrievalResult
from app.retrieval.types import SearchFilters, SearchResult
from tests.conftest import FakeAnswerGenerator, Identity

DOC = uuid.UUID(int=42)
CHUNK = uuid.UUID(int=43)

PASSAGE = ContextBlock(
    document_id=DOC,
    document_title="Travel Policy",
    filename="travel.pdf",
    page_start=3,
    page_end=3,
    heading_path=["Travel Policy", "Hotels"],
    chunk_ids=[CHUNK],
    text="Hotels are reimbursed up to 150 EUR per night.",
    score=0.98,
    token_count=12,
    segments=[ContextSegment("Hotels are reimbursed up to 150 EUR per night.", CHUNK, 3, 3)],
)


class FakeRetriever:
    def __init__(self, blocks: list[ContextBlock]) -> None:
        self.blocks = blocks
        self.calls: list[tuple[Principal, str, SearchFilters]] = []

    async def retrieve(
        self,
        principal: Principal,
        query: str,
        *,
        top_k: int,
        token_budget: int,
        filters: SearchFilters,
    ) -> RetrievalResult:
        self.calls.append((principal, query, filters))
        search = SearchResult(
            hits=[],
            keyword=[],
            vector=[],
            fused=[],
            degraded=[],
            embedding_model="fake",
            timings_ms={"total": 12.5},
            not_found=not self.blocks,
        )
        return RetrievalResult(blocks=self.blocks, search=search)


@pytest.fixture
def retriever(app: FastAPI) -> FakeRetriever:
    fake = FakeRetriever([PASSAGE])
    app.dependency_overrides[get_retriever] = lambda: fake
    return fake


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for raw in re.split(r"\r?\n\r?\n", body.strip()):
        name = data = None
        for line in raw.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if name and data:
            events.append((name, json.loads(data)))
    return events


async def new_conversation(client: httpx.AsyncClient) -> str:
    response = await client.post("/chat", json={})
    assert response.status_code == 201, response.text
    conversation_id: str = response.json()["id"]
    return conversation_id


async def ask(
    client: httpx.AsyncClient, conversation_id: str, question: str, **body: Any
) -> list[tuple[str, dict[str, Any]]]:
    response = await client.post(
        f"/chat/{conversation_id}/messages", json={"content": question, **body}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


async def test_answer_streams_sources_text_citations_then_done(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    conversation_id = await new_conversation(client)
    events = await ask(
        client, conversation_id, "What is the hotel limit?", filters={"file_types": ["pdf"]}
    )

    assert [name for name, _ in events] == ["sources", "delta", "delta", "citation", "done"]
    sources = events[0][1]
    assert sources["not_found"] is False
    assert sources["sources"][0] | {"chunk_ids": None} == {
        "source": 1,
        "title": "Travel Policy > Hotels (p. 3)",
        "document_id": str(DOC),
        "document_title": "Travel Policy",
        "filename": "travel.pdf",
        "page_start": 3,
        "page_end": 3,
        "heading_path": ["Travel Policy", "Hotels"],
        "chunk_ids": None,
        "score": 0.98,
    }
    citation = events[3][1]
    assert (citation["index"], citation["source"], citation["page_start"]) == (0, 1, 3)
    assert citation["chunk_ids"] == [str(CHUNK)]
    assert citation["span_start"] == len("According to the policy, ")

    done = events[-1][1]
    assert done["text"] == "According to the policy, Hotels are reimbursed up to 150 EUR per night."
    assert done["spans"] == [{"start": 25, "end": 71, "citations": [0]}]
    assert done["citation_coverage"] > 0.6
    assert done["usage"]["cost_usd"] == 0.0067
    assert done["model"] == "fake-claude"
    assert set(done["timings_ms"]) >= {"retrieval", "first_token", "generation", "total"}
    message_id = done["message_id"]

    # The question reached the retriever with the request's filters.
    assert retriever.calls[0][1] == "What is the hotel limit?"
    assert retriever.calls[0][2].file_types == ("pdf",)
    assert answer_generator.calls[0][0] == "What is the hotel limit?"

    conversation = (await client.get(f"/chat/{conversation_id}")).json()
    assert conversation["title"] == "What is the hotel limit?"
    user_msg, assistant_msg = conversation["messages"]
    assert (user_msg["role"], user_msg["content"]) == ("user", "What is the hotel limit?")
    assert assistant_msg["id"] == message_id
    assert assistant_msg["content"] == done["text"]
    assert assistant_msg["citations"][0]["chunk_ids"] == [str(CHUNK)]
    assert assistant_msg["metadata"]["spans"] == done["spans"]

    with sync_sessions() as s:
        (log,) = s.scalars(select(QueryLog)).all()
    assert str(log.message_id) == message_id
    assert (log.model, log.input_tokens, log.cache_read_tokens) == ("fake-claude", 1200, 640)
    assert log.cost_usd == pytest.approx(0.0067)
    assert log.context_chunk_ids == [str(CHUNK)]
    assert log.not_found is False and log.error is None


async def test_nothing_found_answers_without_calling_the_llm(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    retriever.blocks = []
    conversation_id = await new_conversation(client)
    events = await ask(client, conversation_id, "Are dogs allowed in the office?")

    assert [name for name, _ in events] == ["sources", "delta", "done"]
    assert events[0][1] == {"sources": [], "not_found": True, "degraded": []}
    assert events[1][1] == {"text": NOT_FOUND_ANSWER}
    assert events[-1][1]["not_found"] is True
    assert answer_generator.calls == []
    with sync_sessions() as s:
        (log,) = s.scalars(select(QueryLog)).all()
    assert log.not_found and log.cost_usd == 0 and log.model is None


async def test_llm_failure_streams_an_error_and_keeps_the_partial_answer(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    answer_generator.error = LLMError("llm_rate_limited", "Too many requests.", retryable=True)
    conversation_id = await new_conversation(client)
    events = await ask(client, conversation_id, "What is the hotel limit?")

    assert [name for name, _ in events] == ["sources", "delta", "error"]
    assert events[-1][1] == {
        "code": "llm_rate_limited",
        "message": "Too many requests.",
        "retryable": True,
        "partial": True,
    }
    messages = (await client.get(f"/chat/{conversation_id}")).json()["messages"]
    assert messages[-1]["content"] == "According to the policy, "
    assert messages[-1]["metadata"]["error"] == "llm_rate_limited"
    with sync_sessions() as s:
        (log,) = s.scalars(select(QueryLog)).all()
    assert log.error == "llm_rate_limited"


async def test_conversations_are_private(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    conversation_id = await new_conversation(client)
    with sync_sessions() as s, s.begin():
        bob = User(tenant_id=tenant_and_user[0], email="bob@acme.test", role=Role.ADMIN)
        s.add(bob)
        s.flush()
        identity.user_id = bob.id

    assert (await client.get(f"/chat/{conversation_id}")).status_code == 404
    response = await client.post(f"/chat/{conversation_id}/messages", json={"content": "hi"})
    assert response.status_code == 404
    assert (await client.get("/chat")).json() == []
    assert retriever.calls == []


async def test_question_validation(client: httpx.AsyncClient, retriever: FakeRetriever) -> None:
    conversation_id = await new_conversation(client)
    for body in ({"content": ""}, {"content": "x" * 2001}, {}):
        response = await client.post(f"/chat/{conversation_id}/messages", json=body)
        assert response.status_code == 422
    assert (
        await client.post(f"/chat/{uuid.uuid4()}/messages", json={"content": "hi"})
    ).status_code == 404


async def test_unexpected_failure_still_ends_the_stream_and_saves(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    answer_generator.error = RuntimeError("bug")  # type: ignore[assignment]
    conversation_id = await new_conversation(client)
    events = await ask(client, conversation_id, "What is the hotel limit?")

    assert [name for name, _ in events] == ["sources", "delta", "error"]
    assert events[-1][1]["code"] == "internal_error"
    assert events[-1][1]["partial"] is True
    with sync_sessions() as s:
        (log,) = s.scalars(select(QueryLog)).all()
    assert log.error == "internal_error"
