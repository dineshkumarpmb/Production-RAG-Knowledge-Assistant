"""Conversations and streamed, cited answers.

    POST   /chat                    start a conversation
    GET    /chat                    the user's conversations
    GET    /chat/{id}               a conversation with its messages
    PATCH  /chat/{id}               rename it
    DELETE /chat/{id}               delete it and its messages
    POST   /chat/{id}/messages      ask a question; the answer streams as Server-Sent Events

Context policy: Claude sees the most recent messages verbatim (CHAT_HISTORY_MESSAGES, each
capped at CHAT_HISTORY_MAX_CHARS) plus a running summary of everything older. Once more than
CHAT_SUMMARY_AFTER_MESSAGES messages are unsummarised, the older ones are folded into the
summary after the answer has been sent. A follow-up question is rewritten into a standalone
search query first, so retrieval works for "what about contractors?".

If the client disconnects mid-answer, the upstream Claude stream is cancelled and the partial
answer is still saved, with the tokens used so far, marked `cancelled`.
"""

import json
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Any

import anyio
import structlog
from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette.sse import EventSourceResponse

from app.api.deps import CurrentPrincipal, DbSession, SettingsDep, rate_limit
from app.api.routes.search import FiltersIn, RetrieverDep
from app.core.audit import audit
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError, request_id_var
from app.db.access import Principal, can_read
from app.db.models import Conversation, Document, Message, MessageRole, QueryLog
from app.db.session import get_async_sessionmaker
from app.generation.claude import (
    AnswerGenerator,
    ClaudeAnswerGenerator,
    LLMError,
    create_client,
)
from app.generation.conversation import ConversationHelper
from app.generation.prompts import HistoryMessage
from app.generation.service import AnswerOutcome, AnswerRun

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_TITLE_LENGTH = 80


@lru_cache
def _claude_generator() -> ClaudeAnswerGenerator:
    settings = get_settings()
    return ClaudeAnswerGenerator(create_client(settings), settings)


def get_answer_generator() -> AnswerGenerator:
    return _claude_generator()


@lru_cache
def _conversation_helper() -> ConversationHelper:
    settings = get_settings()
    return ConversationHelper(create_client(settings), settings)


def get_conversation_helper() -> ConversationHelper | None:
    return _conversation_helper()


GeneratorDep = Annotated[AnswerGenerator, Depends(get_answer_generator)]
HelperDep = Annotated[ConversationHelper | None, Depends(get_conversation_helper)]
SessionsDep = Annotated[async_sessionmaker[AsyncSession], Depends(get_async_sessionmaker)]


class ConversationCreate(BaseModel):
    title: str | None = Field(None, max_length=200)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(validation_alias="meta", serialization_alias="metadata")
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class AskRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    filters: FiltersIn = Field(default_factory=FiltersIn)


async def _own_conversation(
    db: AsyncSession, principal: Principal, conversation_id: uuid.UUID
) -> Conversation:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != principal.user_id:
        raise NotFoundError("Conversation")
    return conversation


_REDACTED = {"redacted": True, "reason": "You no longer have access to this document."}


async def _redact_revoked(
    db: AsyncSession, principal: Principal, messages: list[Message]
) -> list[MessageOut]:
    """Citations and sources of documents the user can no longer read (unshared, deleted) are
    replaced by a placeholder: an old answer must not keep showing their text or titles."""
    doc_ids = {
        uuid.UUID(ref["document_id"])
        for m in messages
        for ref in (*m.citations, *m.meta.get("sources", []))
        if ref.get("document_id")
    }
    readable = (
        set(
            await db.scalars(
                select(Document.id).where(Document.id.in_(doc_ids), can_read(principal))
            )
        )
        if doc_ids
        else set()
    )

    def clean(ref: dict[str, Any]) -> dict[str, Any]:
        if ref.get("document_id") and uuid.UUID(ref["document_id"]) not in readable:
            return {"source": ref.get("source"), **_REDACTED}
        return ref

    out = []
    for m in messages:
        item = MessageOut.model_validate(m)
        item.citations = [clean(c) for c in m.citations]
        if "sources" in item.metadata:
            item.metadata = {**item.metadata, "sources": [clean(x) for x in m.meta["sources"]]}
        out.append(item)
    return out


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate, db: DbSession, principal: CurrentPrincipal
) -> ConversationOut:
    conversation = Conversation(
        tenant_id=principal.tenant_id, user_id=principal.user_id, title=body.title or ""
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return ConversationOut.model_validate(conversation)


@router.get("")
async def list_conversations(
    db: DbSession,
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationOut]:
    rows = await db.scalars(
        select(Conversation)
        .where(Conversation.user_id == principal.user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id)
        .limit(limit)
        .offset(offset)
    )
    return [ConversationOut.model_validate(c) for c in rows]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID, db: DbSession, principal: CurrentPrincipal
) -> ConversationDetail:
    conversation = await _own_conversation(db, principal, conversation_id)
    messages = list(
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at, Message.id)
        )
    )
    return ConversationDetail(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=await _redact_revoked(db, principal, messages),
    )


@router.patch("/{conversation_id}")
async def rename_conversation(
    conversation_id: uuid.UUID,
    body: ConversationUpdate,
    db: DbSession,
    principal: CurrentPrincipal,
) -> ConversationOut:
    conversation = await _own_conversation(db, principal, conversation_id)
    conversation.title = body.title.strip()
    await db.commit()
    await db.refresh(conversation)
    return ConversationOut.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, db: DbSession, principal: CurrentPrincipal
) -> Response:
    """Delete a conversation and its messages. Query logs are kept (unlinked) for cost and
    quality reporting."""
    conversation = await _own_conversation(db, principal, conversation_id)
    await db.execute(
        update(QueryLog)
        .where(QueryLog.conversation_id == conversation.id)
        .values(conversation_id=None, message_id=None)
    )
    await db.execute(delete(Message).where(Message.conversation_id == conversation.id))
    await db.delete(conversation)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _history(messages: Sequence[Message], max_chars: int) -> list[HistoryMessage]:
    turns = []
    for m in messages:
        text = m.content.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + " [...]"
        turns.append(HistoryMessage("user" if m.role == MessageRole.USER else "assistant", text))
    return turns


@router.post(
    "/{conversation_id}/messages",
    dependencies=[Depends(rate_limit("chat", "rate_limit_chat_per_minute"))],
)
async def ask(
    conversation_id: uuid.UUID,
    body: AskRequest,
    request: Request,
    db: DbSession,
    principal: CurrentPrincipal,
    retriever: RetrieverDep,
    generator: GeneratorDep,
    helper: HelperDep,
    sessions: SessionsDep,
    settings: SettingsDep,
) -> EventSourceResponse:
    """Answer a question from the user's documents, streamed as Server-Sent Events:
    `sources`, then `delta` / `citation` (repeated), then `done` (or `error`)."""
    conversation = await _own_conversation(db, principal, conversation_id)
    earlier = list(
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at, Message.id)
        )
    )
    history = _history(earlier[conversation.summarized_messages :], settings.chat_history_max_chars)
    summary = conversation.summary
    if not conversation.title:
        conversation.title = body.content[:_TITLE_LENGTH]
    db.add(Message(conversation_id=conversation.id, role=MessageRole.USER, content=body.content))
    audit(
        db,
        "chat.query",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        target_type="conversation",
        target_id=conversation.id,
        request=request,
    )
    await db.commit()

    run = AnswerRun(
        retriever,
        generator,
        principal,
        body.content,
        filters=body.filters.to_filters(),
        top_k=settings.rerank_top_k,
        token_budget=settings.context_token_budget,
        history=history,
        summary=summary,
        helper=helper,
        request_id=request_id_var.get(),
    )
    message_id = uuid.uuid4()
    tenant_id, user_id = principal.tenant_id, principal.user_id

    async def save() -> None:
        await _save_answer(
            sessions, run.outcome, message_id, conversation_id, tenant_id=tenant_id, user_id=user_id
        )

    async def stream() -> AsyncIterator[dict[str, str]]:
        # Runs after the request's DB session has closed, so it uses its own sessions.
        try:
            async for name, payload in run.events():
                if name == "done":
                    payload = {**payload, "message_id": str(message_id)}
                yield {"event": name, "data": json.dumps(payload)}
        except anyio.get_cancelled_exc_class():
            # The client went away. Leaving the generator closed the upstream Claude stream
            # (no more tokens are billed); keep what was answered and what it cost.
            run.outcome.cancelled = True
            run.outcome.error = run.outcome.error or "client_disconnected"
            with anyio.CancelScope(shield=True):
                await save()
            log.info("answer.cancelled", message_id=str(message_id), partial=bool(run.outcome.text))
            raise
        except Exception:
            # Headers are already sent, so an HTTP error is no longer possible: report the
            # failure as an event and still save what was answered so far.
            log.exception("answer.crashed", message_id=str(message_id))
            run.outcome.error = "internal_error"
            error = {
                "code": "internal_error",
                "message": "Something went wrong while answering.",
                "retryable": True,
                "partial": bool(run.outcome.text),
                "request_id": run.outcome.request_id,
            }
            yield {"event": "error", "data": json.dumps(error)}
        with anyio.CancelScope(shield=True):  # finish even if the client leaves right now
            await save()
            if helper is not None:
                await _update_summary(sessions, helper, settings, conversation_id, message_id)

    return EventSourceResponse(stream())


async def _save_answer(
    sessions: async_sessionmaker[AsyncSession],
    outcome: AnswerOutcome,
    message_id: uuid.UUID,
    conversation_id: uuid.UUID,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
) -> None:
    async with sessions() as session, session.begin():
        session.add(
            Message(
                id=message_id,
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=outcome.text,
                citations=[c.as_dict() for c in outcome.citations],
                meta=outcome.metadata(),
            )
        )
        await session.flush()
        usage = outcome.usage
        session.add(
            QueryLog(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=message_id,
                query=outcome.question,
                rewritten_query=(
                    outcome.search_query if outcome.search_query != outcome.question else None
                ),
                request_id=outcome.request_id,
                retrieved_chunk_ids=outcome.retrieved_chunk_ids,
                context_chunk_ids=outcome.context_chunk_ids,
                latency_ms=outcome.timings_ms,
                model=outcome.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                cost_usd=outcome.total_cost_usd,  # answer + query rewriting
                stop_reason=outcome.stop_reason,
                not_found=outcome.not_found,
                degraded=outcome.degraded,
                error=outcome.error,
            )
        )
        conversation = await session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.updated_at = datetime.now(conversation.created_at.tzinfo)
    log.info(
        "answer.saved",
        message_id=str(message_id),
        cost_usd=outcome.total_cost_usd,
        stop_reason=outcome.stop_reason,
        cancelled=outcome.cancelled,
        not_found=outcome.not_found,
        error=outcome.error,
    )


async def _update_summary(
    sessions: async_sessionmaker[AsyncSession],
    helper: ConversationHelper,
    settings: Settings,
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
) -> None:
    """Fold messages older than the verbatim window into the running summary, once enough have
    piled up. Best effort: on failure the history simply stays longer until the next turn."""
    async with sessions() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            return
        covered, previous = conversation.summarized_messages, conversation.summary
        pending = list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
                .offset(covered)
            )
        )
    if len(pending) <= settings.chat_summary_after_messages:
        return
    fold = pending[: len(pending) - settings.chat_history_messages]
    try:
        result = await helper.summarize(previous, _history(fold, settings.chat_history_max_chars))
    except LLMError as exc:
        log.warning("summary.failed", code=exc.code, conversation_id=str(conversation_id))
        return

    async with sessions() as session, session.begin():
        conversation = await session.get(Conversation, conversation_id, with_for_update=True)
        if conversation is None or conversation.summarized_messages != covered:
            return  # deleted, or another request summarised concurrently
        conversation.summary = result.text
        conversation.summarized_messages = covered + len(fold)
        await session.execute(
            update(QueryLog)
            .where(QueryLog.message_id == message_id)
            .values(cost_usd=QueryLog.cost_usd + result.usage.cost_usd)
        )
    log.info("summary.updated", conversation_id=str(conversation_id), folded=len(fold))
