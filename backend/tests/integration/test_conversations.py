"""Multi-turn chat: follow-up rewriting, history and summaries, conversation management, and
what happens when the client disconnects mid-answer (SQLite; fake retrieval and LLMs)."""

import asyncio
import json
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Conversation, Message, QueryLog
from app.generation.claude import LLMError
from tests.conftest import FakeAnswerGenerator, FakeConversationHelper
from tests.integration.test_chat_api import FakeRetriever, ask, new_conversation, retriever  # noqa: F401


def logs(sessions: sessionmaker[Session]) -> list[QueryLog]:
    with sessions() as s:
        return list(s.scalars(select(QueryLog).order_by(QueryLog.created_at)))


async def test_follow_up_is_rewritten_for_search_and_the_model_sees_the_history(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,  # noqa: F811
    answer_generator: FakeAnswerGenerator,
    conversation_helper: FakeConversationHelper,
    sync_sessions: sessionmaker[Session],
) -> None:
    conversation_id = await new_conversation(client)
    await ask(client, conversation_id, "What is the hotel limit?")
    assert conversation_helper.rewrites == []  # a first question needs no rewriting
    assert retriever.calls[0][1] == "What is the hotel limit?"

    events = await ask(client, conversation_id, "what about contractors?")
    rewritten = "what about contractors? (about: What is the hotel limit?)"
    assert retriever.calls[1][1] == rewritten
    assert events[0][1]["search_query"] == rewritten

    question, history = conversation_helper.rewrites[0]
    assert question == "what about contractors?"
    assert [(m.role, m.text[:25]) for m in history] == [
        ("user", "What is the hotel limit?"),
        ("assistant", "According to the policy, "),
    ]
    # The answer model gets the user's own words plus the earlier turns.
    assert answer_generator.calls[1][0] == "what about contractors?"
    history_seen, summary_seen = answer_generator.histories[1]
    assert [m.role for m in history_seen] == ["user", "assistant"] and summary_seen is None

    done = events[-1][1]
    assert (done["search_query"], done["history_messages"]) == (rewritten, 2)
    first, second = logs(sync_sessions)
    assert first.rewritten_query is None
    assert (second.query, second.rewritten_query) == ("what about contractors?", rewritten)
    assert second.cost_usd == pytest.approx(0.0067 + 0.00001)  # answer + rewriting
    assert second.request_id


async def test_when_rewriting_fails_the_question_is_searched_as_asked(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,  # noqa: F811
    conversation_helper: FakeConversationHelper,
) -> None:
    conversation_id = await new_conversation(client)
    await ask(client, conversation_id, "What is the hotel limit?")
    conversation_helper.error = LLMError("llm_unavailable", "down", retryable=True)
    events = await ask(client, conversation_id, "and for Paris?")

    assert retriever.calls[-1][1] == "and for Paris?"
    assert events[0][1]["degraded"] == ["rewrite_unavailable"]
    assert events[-1][0] == "done"


async def test_older_turns_are_folded_into_a_running_summary(
    app: FastAPI,
    client: httpx.AsyncClient,
    settings: Settings,
    retriever: FakeRetriever,  # noqa: F811
    answer_generator: FakeAnswerGenerator,
    conversation_helper: FakeConversationHelper,
    sync_sessions: sessionmaker[Session],
) -> None:
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"chat_history_messages": 2, "chat_summary_after_messages": 4}
    )
    conversation_id = await new_conversation(client)
    for n in range(1, 4):
        await ask(client, conversation_id, f"Question {n}?")
    # After 3 turns (6 messages, more than 4 unsummarised), the oldest 4 were summarised.
    ((previous, folded),) = conversation_helper.summaries
    assert previous is None
    assert [m.text[:11] for m in folded if m.role == "user"] == ["Question 1?", "Question 2?"]
    with sync_sessions() as s:
        conversation = s.get(Conversation, uuid.UUID(conversation_id))
        assert conversation is not None
        assert (conversation.summary, conversation.summarized_messages) == (
            "Summary of 4 messages",
            4,
        )

    await ask(client, conversation_id, "Question 4?")
    history, summary = answer_generator.histories[-1]
    assert summary == "Summary of 4 messages"
    assert [m.text[:11] for m in history if m.role == "user"] == ["Question 3?"]
    # The summary's cost is added to the turn that triggered it.
    third = logs(sync_sessions)[2]
    assert third.cost_usd == pytest.approx(0.0067 + 0.00001 + 0.00002)


async def test_rename_and_delete_conversations(
    client: httpx.AsyncClient,
    retriever: FakeRetriever,  # noqa: F811
    sync_sessions: sessionmaker[Session],
) -> None:
    conversation_id = await new_conversation(client)
    await ask(client, conversation_id, "What is the hotel limit?")
    renamed = await client.patch(f"/chat/{conversation_id}", json={"title": "Hotels"})
    assert renamed.json()["title"] == "Hotels"
    assert (await client.patch(f"/chat/{conversation_id}", json={"title": ""})).status_code == 422

    assert (await client.delete(f"/chat/{conversation_id}")).status_code == 204
    assert (await client.get(f"/chat/{conversation_id}")).status_code == 404
    (log,) = logs(sync_sessions)  # kept for reporting, unlinked from the deleted chat
    assert (log.conversation_id, log.message_id) == (None, None)
    with sync_sessions() as s:
        assert s.scalars(select(Message)).all() == []


async def disconnect_after_first_delta(app: FastAPI, path: str, body: dict[str, Any]) -> str:
    """Drive the ASGI app directly and hang up once the first piece of answer text arrives."""
    got_text = asyncio.Event()
    request_sent = False
    received: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await got_text.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            received.append(message.get("body", b""))
            if b"event: delta" in b"".join(received):
                got_text.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return b"".join(received).decode()


async def test_client_disconnect_keeps_the_partial_answer_and_its_usage(
    app: FastAPI,
    client: httpx.AsyncClient,
    retriever: FakeRetriever,  # noqa: F811
    answer_generator: FakeAnswerGenerator,
    sync_sessions: sessionmaker[Session],
) -> None:
    conversation_id = await new_conversation(client)
    answer_generator.hang = True  # streams a first piece of text, then waits
    body = await disconnect_after_first_delta(
        app, f"/chat/{conversation_id}/messages", {"content": "What is the hotel limit?"}
    )
    assert "event: delta" in body and "event: done" not in body

    with sync_sessions() as s:
        assistant = s.scalars(select(Message).where(Message.role == "assistant")).one()
        assert assistant.content == "According to the policy, "
        assert assistant.meta["cancelled"] is True
        assert assistant.meta["error"] == "client_disconnected"
    (log,) = logs(sync_sessions)
    assert (log.error, log.input_tokens) == ("client_disconnected", 1200)
    assert log.cost_usd == pytest.approx(0.0048)  # tokens used before the disconnect
