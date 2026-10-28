"""Context assembly: turn the reranked chunks into the passages the model will read.

- Chunks that are next to each other in the same document (consecutive `chunk_index`) are
  merged into one passage, dropping the text the two chunks share through overlap.
- Passages are ordered by their best chunk's relevance.
- Passages are added until the token budget is used up; one that does not fit is skipped so a
  smaller, less relevant one can still be used.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.ingestion.tokens import count_tokens
from app.retrieval.types import SearchHit

_MIN_OVERLAP = 20
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ContextSegment:
    """One paragraph of a passage and the chunk it came from (what a citation points at)."""

    text: str
    chunk_id: uuid.UUID
    page_start: int
    page_end: int


@dataclass
class ContextBlock:
    document_id: uuid.UUID
    document_title: str | None
    filename: str
    page_start: int
    page_end: int
    heading_path: list[str]  # shared by every merged chunk
    chunk_ids: list[uuid.UUID]  # in document order; citations map back through these
    text: str
    score: float  # best relevance among the merged chunks
    token_count: int
    segments: list[ContextSegment] = field(default_factory=list)  # paragraphs, in order


def _relevance(hit: SearchHit) -> float:
    return hit.rerank_score if hit.rerank_score is not None else hit.score


def merge_overlapping(first: str, second: str) -> str:
    """Join two consecutive chunks, keeping the text they share only once.

    The chunker's overlap is whole sentences or paragraphs, so only an overlap of at least
    `_MIN_OVERLAP` characters ending at a word boundary counts; a coincidental match of a few
    characters ("...the" + "e...") must not eat text.
    """
    for size in range(min(len(first), len(second)), _MIN_OVERLAP - 1, -1):
        at_boundary = size == len(second) or second[size].isspace()
        if at_boundary and first.endswith(second[:size]):
            return first + second[size:]
    return f"{first}\n\n{second}"


def _common_prefix(paths: Sequence[list[str]]) -> list[str]:
    prefix = list(paths[0])
    for path in paths[1:]:
        while path[: len(prefix)] != prefix:
            prefix.pop()
    return prefix


def _runs(hits: list[SearchHit]) -> list[list[SearchHit]]:
    """Group one document's hits into runs of consecutive chunks."""
    ordered = sorted(hits, key=lambda h: h.chunk_index)
    runs = [[ordered[0]]]
    for hit in ordered[1:]:
        if hit.chunk_index == runs[-1][-1].chunk_index + 1:
            runs[-1].append(hit)
        else:
            runs.append([hit])
    return runs


def _segments(text: str, run: list[SearchHit]) -> list[ContextSegment]:
    segments = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        if not paragraph.strip():
            continue
        source = next((h for h in run if paragraph.strip() in h.text), run[0])
        segments.append(
            ContextSegment(paragraph, source.chunk_id, source.page_start, source.page_end)
        )
    return segments


def _block(run: list[SearchHit]) -> ContextBlock:
    text = run[0].text
    for hit in run[1:]:
        text = merge_overlapping(text, hit.text)
    return ContextBlock(
        document_id=run[0].document_id,
        document_title=run[0].document_title,
        filename=run[0].filename,
        page_start=min(h.page_start for h in run),
        page_end=max(h.page_end for h in run),
        heading_path=_common_prefix([h.heading_path for h in run]),
        chunk_ids=[h.chunk_id for h in run],
        text=text,
        score=max(_relevance(h) for h in run),
        token_count=count_tokens(text),
        segments=_segments(text, run),
    )


def assemble_context(hits: Sequence[SearchHit], *, token_budget: int) -> list[ContextBlock]:
    by_document: dict[uuid.UUID, list[SearchHit]] = {}
    seen: set[uuid.UUID] = set()
    for hit in hits:
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            by_document.setdefault(hit.document_id, []).append(hit)

    blocks = [_block(run) for doc_hits in by_document.values() for run in _runs(doc_hits)]
    blocks.sort(key=lambda b: -b.score)

    selected: list[ContextBlock] = []
    used = 0
    for block in blocks:
        if used + block.token_count <= token_budget:
            selected.append(block)
            used += block.token_count
    return selected
