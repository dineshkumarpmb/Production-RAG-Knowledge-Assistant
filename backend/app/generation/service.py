"""The answer pipeline: (follow-up? rewrite it) -> retrieve -> (not found? answer without the
LLM) -> stream a cited answer that sees recent turns and a summary of older ones.

Yields `(event, payload)` pairs for Server-Sent Events, in this order:

    sources   the passages the answer will be grounded in (sent before generation starts),
              and the search query used (the rewritten follow-up, when there was one)
    delta     a piece of answer text                                  (repeated)
    citation  a cited passage for the current stretch of text         (repeated)
    reset     discard text and citations so far (refusal / fallback)  (rare)
    done      the final answer: text, cited spans, citations, usage, cost, timings
    error     generation failed; `partial` says whether text was already streamed

`done` and `error` carry the request ID, which also appears in logs and the query log.

After iteration, `AnswerRun.outcome` holds everything needed to store the message and log.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.db.access import Principal
from app.generation.claude import AnswerGenerator, LLMError
from app.generation.conversation import ConversationHelper
from app.generation.prompts import NOT_FOUND_ANSWER, HistoryMessage, source_title
from app.generation.types import (
    Citation,
    CitationAdded,
    Completed,
    Reset,
    Span,
    TextDelta,
    Usage,
    UsageUpdate,
)
from app.retrieval.context import ContextBlock
from app.retrieval.service import HybridRetriever
from app.retrieval.types import SearchFilters

log = structlog.get_logger(__name__)

Event = tuple[str, dict[str, Any]]


@dataclass
class AnswerOutcome:
    question: str
    search_query: str = ""  # what retrieval used: the rewritten follow-up, or the question
    request_id: str | None = None
    history_messages: int = 0
    summary_used: bool = False
    helper_usage: Usage = field(default_factory=Usage)  # query rewriting
    cancelled: bool = False
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    context_chunk_ids: list[str] = field(default_factory=list)
    not_found: bool = False
    degraded: list[str] = field(default_factory=list)
    model: str | None = None
    stop_reason: str | None = None
    fallback_used: bool = False
    truncated: bool = False
    refusal_category: str | None = None
    usage: Usage = field(default_factory=Usage)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def metadata(self) -> dict[str, Any]:
        """What is stored with the assistant message."""
        return {
            "spans": [
                {"start": s.start, "end": s.end, "citations": s.citations} for s in self.spans
            ],
            "sources": self.sources,
            "not_found": self.not_found,
            "degraded": self.degraded,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "fallback_used": self.fallback_used,
            "truncated": self.truncated,
            "refusal_category": self.refusal_category,
            "citation_coverage": citation_coverage(self.text, self.spans),
            "usage": self.usage.__dict__,
            "helper_usage": self.helper_usage.__dict__,
            "search_query": self.search_query,
            "history_messages": self.history_messages,
            "summary_used": self.summary_used,
            "cancelled": self.cancelled,
            "request_id": self.request_id,
            "timings_ms": self.timings_ms,
            "error": self.error,
        }

    @property
    def total_cost_usd(self) -> float:
        return round(self.usage.cost_usd + self.helper_usage.cost_usd, 6)


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        cost_usd=round(a.cost_usd + b.cost_usd, 6),
    )


def citation_coverage(text: str, spans: list[Span]) -> float:
    """Share of the answer's non-whitespace characters inside cited spans (0..1). A low value
    on a long answer means parts of it are not grounded in a cited passage."""
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0
    cited = sum(1 for s in spans for ch in text[s.start : s.end] if not ch.isspace())
    return round(min(cited / total, 1.0), 3)


def _source_payload(n: int, block: ContextBlock) -> dict[str, Any]:
    return {
        "source": n,
        "title": source_title(block),
        "document_id": str(block.document_id),
        "document_title": block.document_title,
        "filename": block.filename,
        "page_start": block.page_start,
        "page_end": block.page_end,
        "heading_path": block.heading_path,
        "chunk_ids": [str(c) for c in block.chunk_ids],
        "score": round(block.score, 4),
    }


class AnswerRun:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        principal: Principal,
        question: str,
        *,
        filters: SearchFilters,
        top_k: int,
        token_budget: int,
        history: list[HistoryMessage] | None = None,
        summary: str | None = None,
        helper: ConversationHelper | None = None,
        request_id: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._principal = principal
        self._filters = filters
        self._top_k = top_k
        self._token_budget = token_budget
        self._history = history or []
        self._summary = summary
        self._helper = helper
        self.outcome = AnswerOutcome(
            question=question,
            search_query=question,
            request_id=request_id,
            history_messages=len(self._history),
            summary_used=bool(summary),
        )

    async def _rewrite(self) -> None:
        """Turn a follow-up into a standalone search query; on failure search the question."""
        out = self.outcome
        if not self._history or self._helper is None:
            return  # a first question is already standalone
        started = time.perf_counter()
        context = self._history
        if self._summary:
            context = [HistoryMessage("assistant", f"(Summary so far) {self._summary}"), *context]
        try:
            result = await self._helper.rewrite(out.question, context)
        except LLMError as exc:
            log.warning("answer.rewrite_unavailable", code=exc.code)
            out.degraded.append("rewrite_unavailable")
        else:
            out.search_query = result.text
            out.helper_usage = add_usage(out.helper_usage, result.usage)
        out.timings_ms["rewrite"] = _ms(started)

    async def events(self) -> AsyncIterator[Event]:
        out = self.outcome
        started = time.perf_counter()

        await self._rewrite()
        retrieval = await self._retriever.retrieve(
            self._principal,
            out.search_query,
            top_k=self._top_k,
            token_budget=self._token_budget,
            filters=self._filters,
        )
        search = retrieval.search
        out.degraded += search.degraded
        out.retrieved_chunk_ids = [str(h.chunk_id) for h in search.hits]
        out.context_chunk_ids = [str(c) for b in retrieval.blocks for c in b.chunk_ids]
        out.sources = [_source_payload(i, b) for i, b in enumerate(retrieval.blocks, start=1)]
        out.not_found = not retrieval.blocks
        out.timings_ms["retrieval"] = search.timings_ms.get("total", 0.0)
        yield (
            "sources",
            {
                "sources": out.sources,
                "not_found": out.not_found,
                "degraded": out.degraded,
                "search_query": out.search_query,
            },
        )

        if out.not_found:
            # Nothing relevant to ground an answer in: say so without calling the LLM.
            out.text = NOT_FOUND_ANSWER
            yield "delta", {"text": out.text}
        else:
            try:
                async for name, payload in self._generate(retrieval.blocks, started):
                    yield name, payload
            except LLMError as exc:
                out.error = exc.code
                out.timings_ms["total"] = _ms(started)
                log.warning("answer.failed", code=exc.code, partial=bool(out.text))
                yield (
                    "error",
                    {
                        "code": exc.code,
                        "message": exc.message,
                        "retryable": exc.retryable,
                        "partial": bool(out.text),
                        "request_id": out.request_id,
                    },
                )
                return

        out.timings_ms["total"] = _ms(started)
        yield "done", self._done_payload()

    async def _generate(self, blocks: list[ContextBlock], started: float) -> AsyncIterator[Event]:
        out = self.outcome
        out.model = self._generator.model
        generation_started = time.perf_counter()
        async for event in self._generator.stream(
            out.question, blocks, self._history, self._summary
        ):
            match event:
                case TextDelta(text=text):
                    if "first_token" not in out.timings_ms:
                        out.timings_ms["first_token"] = _ms(started)
                    out.text += text
                    yield "delta", {"text": text}
                case CitationAdded(index=index, citation=citation, span_start=span_start):
                    yield (
                        "citation",
                        {
                            "index": index,
                            "span_start": span_start,
                            **citation.as_dict(),
                        },
                    )
                case UsageUpdate(usage=usage, model=model):
                    out.usage, out.model = usage, model  # so far: kept if the client leaves
                case Reset(reason=reason, model=model):
                    out.text = ""
                    yield "reset", {"reason": reason, "model": model}
                case Completed() as done:
                    out.text = done.text
                    out.citations = done.citations
                    out.spans = done.spans
                    out.model = done.model
                    out.stop_reason = done.stop_reason
                    out.fallback_used = done.fallback_used
                    out.truncated = done.truncated
                    out.refusal_category = done.refusal_category
                    out.usage = done.usage
        out.timings_ms["generation"] = _ms(generation_started)

    def _done_payload(self) -> dict[str, Any]:
        out = self.outcome
        return {
            "text": out.text,
            "citations": [c.as_dict() for c in out.citations],
            **out.metadata(),
        }


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
