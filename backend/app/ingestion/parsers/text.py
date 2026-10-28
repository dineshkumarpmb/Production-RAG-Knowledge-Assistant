"""Plain text and Markdown. The whole file is a single page."""

import codecs
import re
from pathlib import Path

from app.ingestion.parsers.base import Block, Page, ParsedDocument, ParseError

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode text bytes, returning (text, encoding). Rejects binary content."""
    try:
        if data.startswith(codecs.BOM_UTF8):
            return data[len(codecs.BOM_UTF8) :].decode("utf-8"), "utf-8-sig"
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return data.decode("utf-16"), "utf-16"
    except UnicodeDecodeError as exc:
        raise ParseError("The text file has an invalid encoding.") from exc
    if b"\x00" in data[:8192]:
        raise ParseError("The file looks binary, not text.")
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # Legacy Windows files are the common non-UTF-8 case.
        return data.decode("cp1252", errors="replace"), "cp1252"


def _paragraph_blocks(text: str) -> list[Block]:
    return [Block(p) for p in re.split(r"\n[ \t]*\n", text) if p.strip()]


def _markdown_blocks(text: str) -> list[Block]:
    """Split Markdown into heading and paragraph blocks; fenced code blocks stay whole."""
    blocks: list[Block] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in current):
            blocks.append(Block("\n".join(current)))
        current.clear()

    for line in text.split("\n"):
        if _FENCE.match(line):
            if not in_fence:
                flush()
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            continue
        if in_fence:
            current.append(line)
            continue
        if heading := _ATX_HEADING.match(line):
            flush()
            blocks.append(Block(heading.group(2), heading_level=len(heading.group(1))))
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def parse_text(path: Path, *, markdown: bool) -> ParsedDocument:
    text, encoding = decode_text(path.read_bytes())
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _markdown_blocks(text) if markdown else _paragraph_blocks(text)
    return ParsedDocument(pages=[Page(1, blocks)], page_mode="single", extra={"encoding": encoding})
