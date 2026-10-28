"""Structure-aware chunking.

1. Every heading starts a new section; chunks never cross sections, and a heading is kept with
   the text that follows it.
2. Inside a section, paragraphs are packed into chunks of about `target_tokens`. Paragraphs that
   are too long on their own are split by line, then by sentence, then by words.
3. Each new chunk within a section repeats the last ~`overlap_ratio` of the previous one.

Chunks may span pages. Their text is copied verbatim from the page text, so `char_start` /
`char_end` locate the chunk in `page_start` / `page_end`, which the UI uses to highlight it.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby

from app.ingestion.cleaning import PageContent
from app.ingestion.tokens import count_tokens

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
# Finer and finer split points for oversized paragraphs: lines, sentences, words.
_SPLITTERS = (
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?;:])\s+"),
    re.compile(r"\s+"),
)
_CHARS_PER_TOKEN = 4  # last resort for "words" longer than a whole chunk (e.g. base64 blobs)


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 500
    overlap_ratio: float = 0.15
    contextual_header: bool = True


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    heading_path: tuple[str, ...]
    token_count: int


@dataclass(frozen=True)
class _Unit:
    page: int
    paragraph: int  # running paragraph id, used to rejoin pieces of one paragraph
    start: int
    end: int
    tokens: int
    heading_path: tuple[str, ...]
    is_heading: bool
    section: int


def _spans(text: str, start: int, end: int, splitter: re.Pattern[str]) -> Iterator[tuple[int, int]]:
    """Split text[start:end] at `splitter`, yielding trimmed, non-empty sub-spans."""
    pos = start
    for m in splitter.finditer(text, start, end):
        yield from _trim(text, pos, m.start())
        pos = m.end()
    yield from _trim(text, pos, end)


def _trim(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        yield start, end


def _split_oversized(
    text: str, start: int, end: int, max_tokens: int, level: int = 0
) -> Iterator[tuple[int, int]]:
    if count_tokens(text[start:end]) <= max_tokens:
        yield start, end
        return
    if level == len(_SPLITTERS):
        step = max_tokens * _CHARS_PER_TOKEN
        for pos in range(start, end, step):
            yield pos, min(pos + step, end)
        return
    for s, e in _spans(text, start, end, _SPLITTERS[level]):
        yield from _split_oversized(text, s, e, max_tokens, level + 1)


def _units(pages: Sequence[PageContent], max_tokens: int) -> Iterator[_Unit]:
    path: list[tuple[int, str]] = []  # (level, heading text)
    section = paragraph = 0
    for page in pages:
        headings = {h.char_offset: h for h in page.headings}
        for start, end in _spans(page.text, 0, len(page.text), _PARAGRAPH_BREAK):
            paragraph += 1
            if (heading := headings.get(start)) is not None:
                path = [(lvl, t) for lvl, t in path if lvl < heading.level]
                path.append((heading.level, heading.text))
                section += 1
                tokens = count_tokens(page.text[start:end])
                yield _Unit(page.number, paragraph, start, end, tokens, _names(path), True, section)
                continue
            for s, e in _split_oversized(page.text, start, end, max_tokens):
                tokens = count_tokens(page.text[s:e])
                yield _Unit(page.number, paragraph, s, e, tokens, _names(path), False, section)


def _names(path: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(text for _, text in path)


def _overlap_tail(units: list[_Unit], budget: int) -> list[_Unit]:
    tail: list[_Unit] = []
    used = 0
    for unit in reversed(units):
        if unit.is_heading or used + unit.tokens > budget:
            break
        tail.insert(0, unit)
        used += unit.tokens
    return tail


def _build(index: int, units: list[_Unit], page_text: dict[int, str]) -> Chunk:
    # Pieces of the same paragraph are re-joined from the original text (keeping the original
    # spacing); separate paragraphs are joined with a blank line, as in the page text.
    parts = []
    for (page, _), group in groupby(units, key=lambda u: (u.page, u.paragraph)):
        pieces = list(group)
        parts.append(page_text[page][pieces[0].start : pieces[-1].end])
    text = "\n\n".join(parts)
    body = next((u for u in units if not u.is_heading), units[-1])
    return Chunk(
        index=index,
        text=text,
        page_start=units[0].page,
        page_end=units[-1].page,
        char_start=units[0].start,
        char_end=units[-1].end,
        heading_path=body.heading_path,
        token_count=count_tokens(text),
    )


def chunk_pages(pages: Sequence[PageContent], config: ChunkingConfig) -> list[Chunk]:
    page_text = {p.number: p.text for p in pages}
    overlap_budget = int(config.target_tokens * config.overlap_ratio)

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    tokens = 0
    has_new_body = False  # body text added since the last emitted chunk
    has_new_units = False

    def emit() -> None:
        chunks.append(_build(len(chunks), current, page_text))

    for unit in _units(pages, config.target_tokens):
        if has_new_body and unit.is_heading:
            emit()
            current, tokens, has_new_body, has_new_units = [], 0, False, False
        elif has_new_body and tokens + unit.tokens > config.target_tokens:
            emit()
            current = _overlap_tail(current, overlap_budget)
            tokens = sum(u.tokens for u in current)
            has_new_body = has_new_units = False
        current.append(unit)
        tokens += unit.tokens
        has_new_units = True
        has_new_body = has_new_body or not unit.is_heading

    if current and has_new_units:
        emit()
    return chunks


def with_source_header(title: str | None, heading_path: Sequence[str], text: str) -> str:
    """Prefix text with where it comes from: "Title > Section > Subsection"."""
    header = " > ".join(p for p in (title, *heading_path) if p)
    return f"{header}\n\n{text}" if header else text


def embedding_input(chunk: Chunk, title: str | None, *, contextual_header: bool) -> str:
    """The text actually embedded: the chunk, optionally prefixed with where it comes from."""
    if not contextual_header:
        return chunk.text
    return with_source_header(title, chunk.heading_path, chunk.text)
