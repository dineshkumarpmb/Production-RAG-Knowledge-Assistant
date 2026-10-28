from pathlib import Path

from app.core.config import Settings
from app.db.models import FileType
from app.ingestion.parsers.base import ParsedDocument, ParseError
from app.ingestion.parsers.docx import parse_docx
from app.ingestion.parsers.pdf import parse_pdf
from app.ingestion.parsers.text import parse_text

__all__ = ["ParseError", "ParsedDocument", "parse_file"]


def parse_file(path: Path, file_type: str, settings: Settings) -> ParsedDocument:
    match FileType(file_type):
        case FileType.PDF:
            return parse_pdf(path)
        case FileType.DOCX:
            return parse_docx(path, virtual_page_chars=settings.docx_virtual_page_chars)
        case FileType.TXT:
            return parse_text(path, markdown=False)
        case FileType.MD:
            return parse_text(path, markdown=True)
