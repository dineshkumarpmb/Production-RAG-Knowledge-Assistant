from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ingestion.cleaning import clean_pages
from app.ingestion.parsers import ParseError
from app.ingestion.parsers.docx import parse_docx
from app.ingestion.parsers.pdf import parse_pdf, parse_pdf_date
from app.ingestion.parsers.text import decode_text, parse_text
from tests import files


def page_texts(path: Path) -> list[str]:
    pages, _ = clean_pages(parse_pdf(path).pages)
    return [p.text for p in pages]


class TestPdf:
    def test_pages_metadata_and_headings(self, tmp_path: Path) -> None:
        parsed = parse_pdf(files.handbook_pdf(tmp_path / "h.pdf", pages=4))
        assert parsed.page_mode == "physical"
        assert len(parsed.pages) == 4
        assert parsed.title == "ACME Employee Handbook"
        assert parsed.author == "HR Team"
        assert parsed.created_at == datetime(2024, 1, 15, 8, 30, tzinfo=UTC)
        assert parsed.extra["subject"] == "Policies"

        pages, removed = clean_pages(parsed.pages)
        assert removed == 8  # header + footer on each of 4 pages
        assert pages[1].headings[0].text == "Section 2: Policy Area 2"
        assert pages[1].headings[0].level == 1
        assert pages[1].headings[0].char_offset == 0
        assert "(page 2)" in pages[1].text
        assert "ACME Corp Employee Handbook" not in pages[1].text
        assert "Page 2 of 4" not in pages[1].text

    def test_two_column_reading_order(self, tmp_path: Path) -> None:
        (text,) = page_texts(files.two_column_pdf(tmp_path / "cols.pdf"))
        order = ["Full Width Title", "Left one", "Left two", "Right one", "Right two"]
        positions = [text.index(s) for s in order]
        assert positions == sorted(positions)

    def test_dehyphenation(self, tmp_path: Path) -> None:
        (text,) = page_texts(files.hyphenated_pdf(tmp_path / "hy.pdf"))
        assert text == "This is an example of a hyphenated line and the COVID-19 rules."

    def test_outline_headings_preferred(self, tmp_path: Path) -> None:
        parsed = parse_pdf(files.outline_pdf(tmp_path / "toc.pdf"))
        pages, _ = clean_pages(parsed.pages)
        assert parsed.extra["heading_source"] == "outline"
        assert [(h.text, h.level) for h in pages[1].headings] == [("Scope", 2)]

    def test_empty_pdf_has_no_text(self, tmp_path: Path) -> None:
        pages, _ = clean_pages(parse_pdf(files.empty_pdf(tmp_path / "e.pdf")).pages)
        assert [p.text for p in pages] == ["", ""]

    def test_corrupt_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.7\n this is not really a pdf")
        with pytest.raises(ParseError, match="could not be opened"):
            parse_pdf(path)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("D:20240131", datetime(2024, 1, 31, tzinfo=UTC)),
            ("D:20240131120000Z", datetime(2024, 1, 31, 12, tzinfo=UTC)),
            (
                "D:20240131120000-05'00'",
                datetime(2024, 1, 31, 12, tzinfo=timezone(-timedelta(hours=5))),
            ),
            ("garbage", None),
            ("D:20241399", None),
            (None, None),
        ],
    )
    def test_pdf_dates(self, raw: str | None, expected: datetime | None) -> None:
        assert parse_pdf_date(raw) == expected


class TestDocx:
    def test_headings_tables_and_explicit_break(self, tmp_path: Path) -> None:
        parsed = parse_docx(files.sample_docx(tmp_path / "t.docx"), virtual_page_chars=3000)
        assert parsed.title == "Travel Policy"
        assert parsed.author == "Finance"
        assert parsed.page_mode == "virtual"  # generated file: no Word layout markers
        pages, _ = clean_pages(parsed.pages)
        assert len(pages) == 2
        assert [(h.text, h.level) for h in pages[0].headings] == [
            ("Travel Policy", 1),
            ("Expenses", 2),
        ]
        assert "Hotel | 150 EUR" in pages[0].text
        assert pages[1].text.startswith("Approvals")
        heading = pages[0].headings[1]
        assert pages[0].text[heading.char_offset :].startswith("Expenses")

    def test_virtual_pages_split_long_documents(self, tmp_path: Path) -> None:
        parsed = parse_docx(files.long_docx(tmp_path / "long.docx"), virtual_page_chars=1000)
        assert len(parsed.pages) > 3
        assert all(sum(len(b.text) for b in p.blocks) <= 1000 for p in parsed.pages)

    def test_corrupt_docx(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.docx"
        path.write_bytes(b"PK\x03\x04 broken zip")
        with pytest.raises(ParseError):
            parse_docx(path, virtual_page_chars=3000)


class TestText:
    def test_markdown_headings_and_code_fences(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text(
            "# Guide\n\nIntro text.\n\n## Setup\n\n```bash\n# not a heading\n\nrun it\n```\n",
            encoding="utf-8",
        )
        parsed = parse_text(path, markdown=True)
        pages, _ = clean_pages(parsed.pages)
        assert parsed.page_mode == "single"
        assert [(h.text, h.level) for h in pages[0].headings] == [("Guide", 1), ("Setup", 2)]
        assert "# not a heading\n\nrun it" in pages[0].text

    def test_plain_text_paragraphs(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_bytes(b"First para.\r\n\r\nSecond   para.")
        pages, _ = clean_pages(parse_text(path, markdown=False).pages)
        assert pages[0].text == "First para.\n\nSecond para."
        assert pages[0].headings == []

    @pytest.mark.parametrize(
        ("data", "encoding"),
        [
            ("héllo".encode(), "utf-8"),
            (b"\xef\xbb\xbfh\xc3\xa9llo", "utf-8-sig"),
            ("héllo".encode("utf-16"), "utf-16"),
            ("héllo".encode("cp1252"), "cp1252"),
        ],
    )
    def test_decode_text(self, data: bytes, encoding: str) -> None:
        assert decode_text(data) == ("héllo", encoding)

    def test_binary_rejected(self) -> None:
        with pytest.raises(ParseError):
            decode_text(b"abc\x00\x01\x02")
