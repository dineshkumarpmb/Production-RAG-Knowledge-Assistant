from app.ingestion.cleaning import build_page_content, clean_text, remove_repeated_edges
from app.ingestion.language import detect_language
from app.ingestion.parsers.base import Block, Page


def test_clean_text_normalises_without_destroying_numbers() -> None:
    raw = "E = mc\u00b2\u00a0\u00a0and the  co\u00adoperation\u200b of \ufb01ve teams\u00b9"
    assert clean_text(raw) == "E = mc\u00b2 and the cooperation of five teams\u00b9"


def test_dehyphenation_keeps_real_compounds() -> None:
    assert clean_text("exam-\nple", reflow=True) == "example"
    assert clean_text("Anglo-\nSaxon", reflow=True) == "Anglo-Saxon"
    assert clean_text("COVID-\n19", reflow=True) == "COVID-19"


def test_reflow_joins_wrapped_lines_but_keeps_indentation_otherwise() -> None:
    assert clean_text("one\ntwo  \n three", reflow=True) == "one two three"
    assert clean_text("def f():\n    return 1") == "def f():\n    return 1"
    assert clean_text("a\n\n\n\nb") == "a\n\nb"
    assert clean_text(" \n\t ") == ""


def make_pages(n: int) -> list[Page]:
    return [
        Page(
            i,
            [
                Block("Confidential - Internal", page_edge=True),
                Block(f"Section {i}", heading_level=1),  # repeats modulo digits, but in the body
                Block(f"Body of page {i} is unique {'x' * i}"),
                Block(f"Page {i}", page_edge=True),
            ],
        )
        for i in range(1, n + 1)
    ]


def test_repeated_headers_and_footers_removed() -> None:
    pages = make_pages(5)
    assert remove_repeated_edges(pages) == 10
    assert [[b.text for b in p.blocks] for p in pages][1] == [
        "Section 2",
        "Body of page 2 is unique xx",
    ]
    assert all(len(p.blocks) == 2 for p in pages)


def test_short_documents_keep_edges() -> None:
    pages = make_pages(2)
    assert remove_repeated_edges(pages) == 0
    assert len(pages[0].blocks) == 4


def test_heading_offsets_point_into_page_text() -> None:
    page = Page(
        1,
        [
            Block("Intro paragraph."),
            Block("A  Heading\nthat wraps", heading_level=2),
            Block(""),
            Block("Body."),
        ],
    )
    content = build_page_content(page)
    assert content.text == "Intro paragraph.\n\nA Heading that wraps\n\nBody."
    (heading,) = content.headings
    assert content.text[heading.char_offset :].startswith(heading.text)


def test_language_detection() -> None:
    assert detect_language("The quick brown fox jumps over the lazy dog near the river.") == "en"
    assert detect_language("Le renard brun rapide saute par-dessus le chien paresseux.") == "fr"
    assert detect_language("too short") is None
