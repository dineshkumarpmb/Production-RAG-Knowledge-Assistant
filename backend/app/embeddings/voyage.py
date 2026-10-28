"""Voyage AI embeddings over its REST API (https://docs.voyageai.com/reference/embeddings-api)."""

import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import httpx
import structlog

from app.embeddings.base import EmbeddingError, EmbeddingResult
from app.ingestion.tokens import count_tokens

log = structlog.get_logger(__name__)

InputType = Literal["document", "query"]

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_INPUTS_PER_REQUEST = 1000  # API limit
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 60.0


class _RateLimiter:
    """Spaces out requests evenly. Per process; a shared Redis limiter can come with scaling."""

    def __init__(self, requests_per_minute: int, sleep: Callable[[float], None]) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            self._sleep(delay)


class VoyageEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dim: int,
        price_per_million_tokens: float,
        base_url: str = "https://api.voyageai.com/v1",
        batch_size: int = 128,
        max_batch_tokens: int = 100_000,
        max_retries: int = 5,
        requests_per_minute: int = 300,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_name = model
        self.dim = dim
        self.price_per_million_tokens = price_per_million_tokens
        self._batch_size = min(batch_size, _MAX_INPUTS_PER_REQUEST)
        self._max_batch_tokens = max_batch_tokens
        self._max_retries = max_retries
        self._sleep = sleep
        self._limiter = _RateLimiter(requests_per_minute, sleep)
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[list[float]] = []
        tokens = 0
        for batch in self._batches(texts):
            result = self._request(batch, "document")
            vectors.extend(result.vectors)
            tokens += result.tokens
        return EmbeddingResult(vectors, tokens)

    def embed_query(self, text: str) -> list[float]:
        return self._request([text], "query").vectors[0]

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            tokens = count_tokens(text)
            if batch and (
                len(batch) >= self._batch_size or batch_tokens + tokens > self._max_batch_tokens
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            yield batch

    def _request(self, inputs: list[str], input_type: InputType) -> EmbeddingResult:
        payload = {
            "input": inputs,
            "model": self.model_name,
            "input_type": input_type,
            "output_dimension": self.dim,
            "truncation": True,
        }
        body = post_with_retries(
            self._client,
            "/embeddings",
            payload,
            max_retries=self._max_retries,
            error=EmbeddingError,
            service="Voyage API",
            sleep=self._sleep,
            before_attempt=self._limiter.wait,
        )
        return self._parse(body, len(inputs))

    def _parse(self, body: dict[str, Any], expected: int) -> EmbeddingResult:
        data = sorted(body["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != expected:
            raise EmbeddingError(
                f"Voyage returned {len(vectors)} embeddings for {expected} inputs."
            )
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"Voyage returned embeddings that are not {self.dim}-dimensional.")
        return EmbeddingResult(vectors, int(body.get("usage", {}).get("total_tokens", 0)))


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return min(float(response.headers["retry-after"]), _BACKOFF_CAP_S)
    except (KeyError, ValueError):
        return None


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or response.reason_phrase)[:200]


def post_with_retries(
    client: httpx.Client,
    path: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
    error: type[Exception],
    service: str,
    sleep: Callable[[float], None] | None = None,
    before_attempt: Callable[[], None] | None = None,
) -> Any:
    """POST JSON, retrying network errors, 408/429/5xx (honouring Retry-After) with jittered
    exponential backoff. Other errors raise `error` at once. Returns the decoded JSON body."""
    pause = sleep or time.sleep
    problem = "no attempt made"
    for attempt in range(max_retries + 1):
        if before_attempt:
            before_attempt()
        retry_after: float | None = None
        try:
            response = client.post(path, json=payload)
        except httpx.TransportError as exc:
            problem = f"network error ({type(exc).__name__})"
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in _RETRYABLE_STATUS:
                raise error(
                    f"{service} rejected the request (HTTP {response.status_code}): "
                    f"{_error_detail(response)}"
                )
            problem = f"HTTP {response.status_code}"
            retry_after = _retry_after(response)

        if attempt == max_retries:
            break
        cap = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2**attempt)
        delay = retry_after if retry_after is not None else random.uniform(0, cap)  # noqa: S311
        log.warning(
            "http.retry",
            service=service,
            attempt=attempt + 1,
            problem=problem,
            delay_s=round(delay, 2),
        )
        pause(delay)

    raise error(f"{service} unavailable after {max_retries + 1} attempts ({problem}).")
