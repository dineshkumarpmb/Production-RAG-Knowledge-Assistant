"""Text clean-up applied to every parsed document before it is stored."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.ingestion.parsers.base import Page

# NFC rather than NFKC: NFKC would turn "x²" into "x2" and "¹" footnote markers into digits,
# which corrupts numbers. Ligatures are the useful part of NFKC, so expand those explicitly.
_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)
_ZERO_WIDTH = re.compile(r"[\u00ad\u200b-\u200d\u2060\ufeff]")  # includes the soft hyphen
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HSPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_HYPHEN_BREAK = re.compile(r"([^\W\d_])-\n[ \t]*([^\W\d_])")
_TRAILING_HYPHEN = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*")
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")
_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class HeadingRef:
    text: str
    level: int
    char_offset: int  # position of the heading within the page text


@dataclass
class PageContent:
    number: int
    text: str
    headings: list[HeadingRef]

    @property
    def char_count(self) -> int:
        return len(self.text)


def _join_hyphenated(match: re.Match[str]) -> str:
    before, after = match.group(1), match.group(2)
    # "exam-\nple" -> "example", but keep "COVID-\n19" style and "Anglo-\nSaxon" compounds.
    return before + after if after.islower() else f"{before}-{after}"


def clean_text(text: str, *, reflow: bool = False) -> str:
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(_join_hyphenated, text)
    if reflow:
        text = _TRAILING_HYPHEN.sub("-", text)  # "COVID-\n19" -> "COVID-19"
        text = _LINE_BREAK.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.lstrip(" \t")
        indent = line[: len(line) - len(stripped)]  # keep indentation (Markdown code, lists)
        lines.append(indent + _HSPACE.sub(" ", stripped).rstrip())
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines))
    return text.strip("\n").rstrip() if text.strip() else ""


def _edge_key(text: str) -> str:
    """Normalise a header/footer line so "Page 3 of 10" and "Page 4 of 10" compare equal."""
    return _DIGITS.sub("#", _WHITESPACE.sub(" ", text.lower())).strip()


def remove_repeated_edges(pages: list[Page], *, min_pages: int = 3, min_ratio: float = 0.6) -> int:
    """Drop running headers/footers: page-margin blocks whose text repeats on most pages.

    Only blocks flagged `page_edge` by the parser are candidates, so numbered headings such as
    "Section 2" in the body are never mistaken for headers. Mutates `pages` in place and
    returns the number of blocks removed.
    """
    if len(pages) < min_pages:
        return 0

    counts: Counter[str] = Counter()
    for page in pages:
        keys = {_edge_key(b.text) for b in page.blocks if b.page_edge}
        counts.update(k for k in keys if k and len(k) <= 200)

    threshold = max(min_pages, min_ratio * len(pages))
    repeated = {key for key, count in counts.items() if count >= threshold}
    if not repeated:
        return 0

    removed = 0
    for page in pages:
        kept = [b for b in page.blocks if not (b.page_edge and _edge_key(b.text) in repeated)]
        removed += len(page.blocks) - len(kept)
        page.blocks = kept
    return removed


def build_page_content(page: Page) -> PageContent:
    """Clean each block and join them with blank lines, recording where headings start."""
    parts: list[str] = []
    headings: list[HeadingRef] = []
    offset = 0
    for block in page.blocks:
        text = clean_text(block.text, reflow=block.reflow or block.is_heading)
        if not text:
            continue
        if parts:
            offset += 2  # the "\n\n" separator
        if block.heading_level is not None:
            headings.append(HeadingRef(text, block.heading_level, offset))
        parts.append(text)
        offset += len(text)
    return PageContent(page.number, "\n\n".join(parts), headings)


def clean_pages(pages: list[Page]) -> tuple[list[PageContent], int]:
    removed = remove_repeated_edges(pages)
    return [build_page_content(p) for p in pages], removed
