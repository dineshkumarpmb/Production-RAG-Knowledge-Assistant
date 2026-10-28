"""Grounded, cited, streamed answers from Claude.

- Model `claude-opus-5-5` by default. Thinking is always on for this model, so `thinking` is
  omitted and depth is set with `output_config.effort` (default `medium`; set explicitly).
- Citations: every passage is a custom-content `document` block with citations enabled.
- Prompt caching: the fixed system prompt carries `cache_control`.
- Refusals: server-side `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) re-runs
  a declined request on the model Anthropic recommends for that refusal category. A final
  `stop_reason == "refusal"` means the whole chain declined.
"""

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any, Protocol

import anthropic
import structlog

from app.core.config import Settings
from app.generation.citations import CitationResolver
from app.generation.pricing import usage_cost
from app.generation.prompts import (
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    HistoryMessage,
    build_messages,
)
from app.generation.types import (
    Citation,
    CitationAdded,
    Completed,
    GenerationEvent,
    Reset,
    Span,
    TextDelta,
    UsageUpdate,
)
from app.retrieval.context import ContextBlock

log = structlog.get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """A generation failure that is safe to show to the user."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def map_error(exc: anthropic.AnthropicError) -> LLMError:
    """Most specific first: retryable (rate limit, overload, network) vs. permanent failures."""
    if isinstance(exc, anthropic.CredentialsError | anthropic.AuthenticationError):
        return LLMError(
            "llm_not_configured", "The Claude API key is missing or invalid.", retryable=False
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMError(
            "llm_not_configured", "The Claude API key lacks permission.", retryable=False
        )
    if isinstance(exc, anthropic.RateLimitError):
        return LLMError(
            "llm_rate_limited", "Too many requests to Claude; try again shortly.", retryable=True
        )
    if isinstance(exc, anthropic.BadRequestError):
        return LLMError(
            "llm_bad_request", f"Claude rejected the request: {exc.message}", retryable=False
        )
    if isinstance(exc, anthropic.APIStatusError):
        retryable = exc.status_code >= 500 or exc.status_code in (408, 409, 429)
        return LLMError(
            "llm_unavailable", f"Claude API error (HTTP {exc.status_code}).", retryable=retryable
        )
    if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
        return LLMError("llm_unavailable", "Could not reach the Claude API.", retryable=True)
    return LLMError("llm_unavailable", "The Claude API request failed.", retryable=True)


def create_client(settings: Settings) -> anthropic.AsyncAnthropic:
    """Credentials: ANTHROPIC_API_KEY setting if given, else the SDK's own resolution
    (ANTHROPIC_API_KEY env var, auth token, or an `ant auth login` profile)."""
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return anthropic.AsyncAnthropic(
        api_key=key, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries
    )


def to_llm_error(exc: Exception) -> LLMError | None:
    """The user-facing error for an SDK failure, or None if `exc` is not one (a real bug)."""
    if isinstance(exc, anthropic.AnthropicError):
        return map_error(exc)
    # With no credentials at all the SDK raises a plain TypeError while building the request
    # headers (seen in the Linux container).
    if isinstance(exc, TypeError) and "authentication method" in str(exc):
        return LLMError("llm_not_configured", "The Claude API key is not set.", retryable=False)
    return None


class AnswerGenerator(Protocol):
    model: str

    def stream(
        self,
        question: str,
        blocks: Sequence[ContextBlock],
        history: Sequence[HistoryMessage] = (),
        summary: str | None = None,
    ) -> AsyncIterator[GenerationEvent]: ...


class _AnswerState:
    """Turns raw stream events into our events, tracking text, cited spans and citations."""

    def __init__(self, resolver: CitationResolver) -> None:
        self._resolver = resolver
        self.model: str | None = None
        self.usage: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.spans: list[Span] = []
        self.citations: list[Citation] = []
        self._citation_keys: dict[tuple[Any, ...], int] = {}

    def on_event(self, event: Any) -> list[GenerationEvent] | dict[str, int]:
        """Our events for one stream event; a dict when it only updated the token counts."""
        if event.type == "message_start":
            self.model = event.message.model
            self.usage = _counts(event.message.usage)
            return dict(self.usage)
        if event.type == "message_delta" and getattr(event, "usage", None) is not None:
            output = getattr(event.usage, "output_tokens", None)
            if output is None:
                return []
            self.usage["output_tokens"] = output
            return dict(self.usage)
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "text":
                self.spans.append(Span(len(self.text), len(self.text)))
            elif block.type == "fallback":
                # A fallback model took over and restarts the answer.
                self.reset()
                return [Reset("fallback", model=getattr(block.to, "model", None))]
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                if not self.spans:
                    self.spans.append(Span(len(self.text), len(self.text)))
                self.text += delta.text
                self.spans[-1].end = len(self.text)
                return [TextDelta(delta.text)]
            if delta.type == "citations_delta":
                return self._add_citation(delta.citation)
        return []

    def _add_citation(self, raw: Any) -> list[GenerationEvent]:
        citation = self._resolver.resolve(raw)
        if citation is None:
            log.warning("generation.unresolvable_citation", citation=str(raw)[:200])
            return []
        key = (citation.source, tuple(citation.chunk_ids), citation.cited_text)
        if key not in self._citation_keys:
            self._citation_keys[key] = len(self.citations)
            self.citations.append(citation)
        index = self._citation_keys[key]
        if not self.spans:
            self.spans.append(Span(len(self.text), len(self.text)))
        span = self.spans[-1]
        if index not in span.citations:
            span.citations.append(index)
        return [CitationAdded(index=index, citation=citation, span_start=span.start)]


class ClaudeAnswerGenerator:
    def __init__(self, client: anthropic.AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.model = settings.llm_model

    def request_params(
        self,
        question: str,
        blocks: Sequence[ContextBlock],
        history: Sequence[HistoryMessage] = (),
        summary: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._settings.llm_max_tokens,
            "system": [
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": build_messages(question, blocks, history, summary),
            "output_config": {"effort": self._settings.llm_effort},
        }
        if self._settings.llm_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    async def stream(
        self,
        question: str,
        blocks: Sequence[ContextBlock],
        history: Sequence[HistoryMessage] = (),
        summary: str | None = None,
    ) -> AsyncIterator[GenerationEvent]:
        state = _AnswerState(CitationResolver(blocks))
        params = self.request_params(question, blocks, history, summary)
        try:
            async with self._client.beta.messages.stream(**params) as stream:
                async for event in stream:
                    result = state.on_event(event)
                    if isinstance(result, dict):
                        model = state.model or self.model
                        usage = usage_cost(_as_usage(result), model, self._settings.llm_prices)
                        yield UsageUpdate(usage=usage, model=model)
                        continue
                    for out in result:
                        yield out
                final = await stream.get_final_message()
        except (anthropic.AnthropicError, TypeError) as exc:
            error = to_llm_error(exc)
            if error is None:
                raise
            log.warning("generation.failed", error=type(exc).__name__, detail=str(exc)[:300])
            raise error from exc
        for out in self._finish(state, final):
            yield out

    def _finish(self, state: _AnswerState, final: Any) -> list[GenerationEvent]:
        events: list[GenerationEvent] = []
        refusal_category = None
        if final.stop_reason == "refusal":
            # Partial output from a declined attempt must not be shown as an answer.
            details = getattr(final, "stop_details", None)
            refusal_category = getattr(details, "category", None)
            state.reset()
            state.text = REFUSAL_ANSWER
            events += [Reset("refusal", model=final.model), TextDelta(REFUSAL_ANSWER)]
            log.info("generation.refused", category=refusal_category, model=final.model)

        usage = getattr(final, "usage", None)
        iterations = getattr(usage, "iterations", None) or []
        events.append(
            Completed(
                text=state.text,
                spans=[s for s in state.spans if s.citations and s.end > s.start],
                citations=state.citations,
                stop_reason=final.stop_reason,
                model=final.model,
                fallback_used=any(it.type == "fallback_message" for it in iterations),
                truncated=final.stop_reason == "max_tokens",
                refusal_category=refusal_category,
                usage=usage_cost(usage, final.model, self._settings.llm_prices),
            )
        )
        return events


_COUNT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _counts(usage: Any) -> dict[str, int]:
    return {f: getattr(usage, f, 0) or 0 for f in _COUNT_FIELDS}


def _as_usage(counts: dict[str, int]) -> Any:
    return SimpleNamespace(**counts, iterations=None)
