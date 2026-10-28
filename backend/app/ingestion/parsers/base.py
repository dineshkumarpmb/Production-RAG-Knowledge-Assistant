from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

PageMode = Literal["physical", "rendered", "virtual", "single"]
"""How page numbers were derived:

- physical: real PDF pages
- rendered: DOCX page breaks recorded by Word's last layout (close to the printed pages)
- virtual:  DOCX split at explicit page breaks and ~N-character boundaries (no layout info)
- single:   TXT / MD, the whole file is page 1
"""


class ParseError(Exception):
    """The file cannot be parsed. The message is shown to the user, so keep it free of internals."""


@dataclass
class Block:
    """A paragraph-like unit of text in reading order."""

    text: str
    heading_level: int | None = None  # None for body text, 1 for top-level headings
    reflow: bool = False  # join hard line breaks (PDF line wraps) during cleaning
    page_edge: bool = False  # in the page margin: a candidate running header/footer

    @property
    def is_heading(self) -> bool:
        return self.heading_level is not None


@dataclass
class Page:
    number: int  # 1-based
    blocks: list[Block]


@dataclass
class ParsedDocument:
    pages: list[Page]
    page_mode: PageMode
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # format-specific metadata
