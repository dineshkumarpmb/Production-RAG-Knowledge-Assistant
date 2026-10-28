"""PDF parsing with PyMuPDF: per-page text in reading order, headings and document metadata.

Headings come from the PDF outline (bookmarks) when it matches the page text; otherwise from
font-size heuristics (text noticeably larger than the body font, or short bold lines).
"""

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pymupdf
import structlog

from app.ingestion.parsers.base import Block, Page, ParsedDocument, ParseError

log = structlog.get_logger(__name__)

_HEADING_SIZE_RATIO = 1.15  # at least 15% larger than body text
_MAX_HEADING_CHARS = 150
_MAX_HEADING_LINES = 3
_MAX_SIZE_LEVELS = 4
_OUTLINE_MIN_MATCH_RATIO = 0.5
_COLUMN_TOLERANCE = 0.05  # fraction of page width a column block may cross the centre line
_MARGIN_RATIO = 0.08  # top/bottom 8% of the page height (~1 inch on A4/Letter)

_PDF_DATE = re.compile(
    r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:([Zz+-])(\d{2})?'?(\d{2})?'?)?"
)


@dataclass
class _RawBlock:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float  # dominant font size, weighted by characters
    bold: bool
    line_count: int
    in_margin: bool  # inside the top/bottom margin, where running headers and footers live
    heading_level: int | None = None


def parse_pdf_date(value: str | None) -> datetime | None:
    """Parse a PDF date string such as "D:20240131093000+01'00'"."""
    if not value or not (m := _PDF_DATE.match(value.strip())):
        return None
    year, month, day, hour, minute, second, sign, tz_h, tz_m = m.groups()
    tz: timezone = UTC
    if sign in ("+", "-"):
        offset = timedelta(hours=int(tz_h or 0), minutes=int(tz_m or 0))
        tz = timezone(offset if sign == "+" else -offset)
    try:
        return datetime(
            int(year),
            int(month or 1),
            int(day or 1),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
            tzinfo=tz,
        )
    except ValueError:
        return None


def _make_block(lines: list[dict[str, Any]], page_height: float) -> _RawBlock | None:
    texts: list[str] = []
    size_weight: Counter[float] = Counter()
    bold_chars = total_chars = 0
    for line in lines:
        spans = line["spans"]
        texts.append("".join(s["text"] for s in spans))
        for span in spans:
            n = len(span["text"].strip())
            total_chars += n
            size_weight[round(span["size"] * 2) / 2] += n
            if span["flags"] & pymupdf.TEXT_FONT_BOLD or "bold" in span["font"].lower():
                bold_chars += n
    text = "\n".join(texts)
    if not text.strip():
        return None
    x0 = min(line["bbox"][0] for line in lines)
    y0 = min(line["bbox"][1] for line in lines)
    x1 = max(line["bbox"][2] for line in lines)
    y1 = max(line["bbox"][3] for line in lines)
    return _RawBlock(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        size=size_weight.most_common(1)[0][0] if size_weight else 0.0,
        bold=total_chars > 0 and bold_chars / total_chars > 0.5,
        line_count=len(texts),
        in_margin=y1 <= page_height * _MARGIN_RATIO or y0 >= page_height * (1 - _MARGIN_RATIO),
    )


def _split_columns(lines: list[dict[str, Any]], page_width: float) -> list[list[dict[str, Any]]]:
    """MuPDF sometimes merges lines from two columns that share a baseline into one block.

    If every line of a block sits clearly in one half of the page, split it per column.
    """
    mid, tolerance = page_width / 2, page_width * _COLUMN_TOLERANCE
    left = [ln for ln in lines if ln["bbox"][2] <= mid + tolerance]
    right = [ln for ln in lines if ln["bbox"][0] >= mid - tolerance]
    if left and right and len(left) + len(right) == len(lines):
        return [left, right]
    return [lines]


def _extract_blocks(page: Any) -> list[_RawBlock]:
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    blocks: list[_RawBlock] = []
    for block in data["blocks"]:
        if block.get("type") != 0 or not block["lines"]:  # type 1 = image
            continue
        for lines in _split_columns(block["lines"], page.rect.width):
            if raw := _make_block(lines, page.rect.height):
                blocks.append(raw)
    return blocks


def order_blocks(blocks: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Put blocks in reading order, handling two-column layouts.

    Full-width blocks (titles, figures spanning both columns) split the page into bands;
    inside each band the left column is read before the right one.
    """
    mid = page_width / 2
    tolerance = page_width * _COLUMN_TOLERANCE

    def column(b: _RawBlock) -> int:  # 0 = left, 1 = right, 2 = spans both
        if b.x1 <= mid + tolerance:
            return 0
        if b.x0 >= mid - tolerance:
            return 1
        return 2

    columns = [column(b) for b in blocks]
    if columns.count(0) < 2 or columns.count(1) < 2:
        return sorted(blocks, key=lambda b: (round(b.y0), b.x0))

    full_width = sorted(
        (b for b, c in zip(blocks, columns, strict=True) if c == 2), key=lambda b: b.y0
    )
    in_columns = [(b, c) for b, c in zip(blocks, columns, strict=True) if c != 2]

    ordered: list[_RawBlock] = []
    band_top = float("-inf")
    for separator in [*full_width, None]:
        band_bottom = separator.y0 if separator else float("inf")
        band = [(b, c) for b, c in in_columns if band_top <= b.y0 < band_bottom]
        for col in (0, 1):
            ordered.extend(sorted((b for b, c in band if c == col), key=lambda b: b.y0))
        if separator:
            ordered.append(separator)
            band_top = separator.y0
    return ordered


def _is_heading_candidate(block: _RawBlock) -> bool:
    text = block.text.strip()
    return (
        0 < len(text) <= _MAX_HEADING_CHARS
        and block.line_count <= _MAX_HEADING_LINES
        and any(ch.isalpha() for ch in text)
    )


def _apply_font_headings(pages: list[list[_RawBlock]], body_size: float) -> None:
    all_blocks = [b for page in pages for b in page]
    large_sizes = sorted(
        {
            b.size
            for b in all_blocks
            if _is_heading_candidate(b) and b.size >= body_size * _HEADING_SIZE_RATIO
        },
        reverse=True,
    )
    level_by_size = {size: min(i + 1, _MAX_SIZE_LEVELS) for i, size in enumerate(large_sizes)}
    bold_level = min(len(set(level_by_size.values())) + 1, 6)

    for block in all_blocks:
        if not _is_heading_candidate(block):
            continue
        if block.size in level_by_size:
            block.heading_level = level_by_size[block.size]
        elif (
            block.bold
            and block.size >= body_size * 0.95
            and len(block.text.strip()) <= 100
            and not block.text.rstrip().endswith((".", ",", ";"))
        ):
            block.heading_level = bold_level


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _apply_outline_headings(pages: list[list[_RawBlock]], toc: list[list[Any]]) -> float:
    """Mark blocks that match outline entries; returns the share of entries matched."""
    matched = 0
    for level, title, page_number, *_ in toc:
        if not 1 <= page_number <= len(pages) or not (wanted := _normalise(str(title))):
            continue
        for block in pages[page_number - 1]:
            text = _normalise(block.text)
            if block.heading_level is None and (text == wanted or text.startswith(wanted)):
                block.heading_level = min(int(level), 6)
                matched += 1
                break
    return matched / len(toc) if toc else 0.0


def _body_font_size(pages: list[list[_RawBlock]]) -> float:
    weights: Counter[float] = Counter()
    for page in pages:
        for block in page:
            weights[block.size] += len(block.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def parse_pdf(path: Path) -> ParsedDocument:
    try:
        doc: Any = pymupdf.open(path, filetype="pdf")  # PyMuPDF's type hints are incomplete
    except Exception as exc:
        raise ParseError("The PDF could not be opened; the file may be corrupt.") from exc

    with doc:
        if doc.needs_pass:
            raise ParseError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise ParseError("The PDF has no pages.")

        raw_pages: list[list[_RawBlock]] = []
        unreadable: list[int] = []
        for page in doc:
            try:
                blocks = _extract_blocks(page)
            except Exception:
                log.warning("pdf.page_unreadable", page=page.number + 1, exc_info=True)
                unreadable.append(page.number + 1)
                blocks = []
            raw_pages.append(order_blocks(blocks, page.rect.width))

        toc = doc.get_toc(simple=True)
        heading_source = "font"
        if toc and _apply_outline_headings(raw_pages, toc) >= _OUTLINE_MIN_MATCH_RATIO:
            heading_source = "outline"
        else:
            for block in (b for page in raw_pages for b in page):
                block.heading_level = None
            _apply_font_headings(raw_pages, _body_font_size(raw_pages))

        meta = doc.metadata or {}
        pages = [
            Page(
                number=i + 1,
                blocks=[
                    Block(
                        b.text,
                        heading_level=b.heading_level,
                        reflow=True,
                        page_edge=b.in_margin,
                    )
                    for b in blocks
                ],
            )
            for i, blocks in enumerate(raw_pages)
        ]

    extra: dict[str, Any] = {
        "heading_source": heading_source,
        "outline_entries": len(toc),
        "subject": meta.get("subject") or None,
        "keywords": meta.get("keywords") or None,
        "creator": meta.get("creator") or None,
        "producer": meta.get("producer") or None,
        "modified_at": (d.isoformat() if (d := parse_pdf_date(meta.get("modDate"))) else None),
    }
    if unreadable:
        extra["unreadable_pages"] = unreadable

    return ParsedDocument(
        pages=pages,
        page_mode="physical",
        title=(meta.get("title") or "").strip() or None,
        author=(meta.get("author") or "").strip() or None,
        created_at=parse_pdf_date(meta.get("creationDate")),
        extra={k: v for k, v in extra.items() if v is not None},
    )
