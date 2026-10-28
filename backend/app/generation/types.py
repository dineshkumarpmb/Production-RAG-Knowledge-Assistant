import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Citation:
    """A cited passage, resolved from Claude's citation back to our document, pages and chunks."""

    source: int  # 1-based number of the context passage (matches the `sources` event)
    document_id: uuid.UUID
    document_title: str | None
    filename: str
    page_start: int
    page_end: int
    heading_path: list[str]
    chunk_ids: list[uuid.UUID]
    cited_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "document_id": str(self.document_id),
            "document_title": self.document_title,
            "filename": self.filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": self.heading_path,
            "chunk_ids": [str(c) for c in self.chunk_ids],
            "cited_text": self.cited_text,
        }


@dataclass
class Span:
    """A stretch of the answer text supported by one or more citations (indices into the list)."""

    start: int
    end: int
    citations: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0  # includes thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


# --- events produced while an answer streams ---------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class CitationAdded:
    index: int  # position in the answer's citation list (repeats reuse the same index)
    citation: Citation
    span_start: int  # where in the answer text the cited span begins


@dataclass(frozen=True)
class Reset:
    """Discard the text and citations streamed so far (a refusal, or a fallback model taking
    over mid-answer, which restarts the answer)."""

    reason: str  # "fallback" | "refusal"
    model: str | None = None


@dataclass(frozen=True)
class UsageUpdate:
    """Tokens used so far: lets an interrupted answer still be costed and logged."""

    usage: Usage
    model: str


@dataclass(frozen=True)
class Completed:
    text: str
    spans: list[Span]
    citations: list[Citation]
    stop_reason: str | None
    model: str
    fallback_used: bool
    truncated: bool
    refusal_category: str | None
    usage: Usage


GenerationEvent = TextDelta | CitationAdded | Reset | UsageUpdate | Completed
