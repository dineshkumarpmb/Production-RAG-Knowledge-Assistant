from itertools import pairwise

from app.ingestion.chunking import Chunk, ChunkingConfig, chunk_pages, embedding_input
from app.ingestion.cleaning import HeadingRef, PageContent
from app.ingestion.tokens import count_tokens


def page(number: int, *blocks: str | tuple[int, str]) -> PageContent:
    """Build a page from paragraphs; (level, text) tuples are headings."""
    parts: list[str] = []
    headings: list[HeadingRef] = []
    offset = 0
    for block in blocks:
        if parts:
            offset += 2
        text = block[1] if isinstance(block, tuple) else block
        if isinstance(block, tuple):
            headings.append(HeadingRef(text, block[0], offset))
        parts.append(text)
        offset += len(text)
    return PageContent(number, "\n\n".join(parts), headings)


def paragraph(n: int, words: int = 40) -> str:
    return " ".join(f"p{n}w{i}" for i in range(words)) + "."


def check_offsets(chunks: list[Chunk], pages: list[PageContent]) -> None:
    by_number = {p.number: p.text for p in pages}
    for c in chunks:
        parts = c.text.split("\n\n")
        first, last = parts[0], parts[-1]
        assert by_number[c.page_start][c.char_start :].startswith(first)
        assert by_number[c.page_end][: c.char_end].endswith(last)


def test_small_document_is_one_chunk() -> None:
    pages = [page(1, (1, "Intro"), "First paragraph.", "Second paragraph.")]
    (chunk,) = chunk_pages(pages, ChunkingConfig())
    assert chunk.text == "Intro\n\nFirst paragraph.\n\nSecond paragraph."
    assert chunk.heading_path == ("Intro",)
    assert (chunk.page_start, chunk.page_end, chunk.char_start) == (1, 1, 0)
    assert chunk.char_end == len(pages[0].text)
    assert chunk.token_count == count_tokens(chunk.text)


def test_headings_start_sections_and_build_paths() -> None:
    pages = [
        page(1, (1, "Policy"), "Policy body.", (2, "Travel"), "Travel body."),
        page(2, (1, "Security"), "Security body."),
    ]
    chunks = chunk_pages(pages, ChunkingConfig())
    assert [c.heading_path for c in chunks] == [
        ("Policy",),
        ("Policy", "Travel"),
        ("Security",),
    ]
    assert [c.text.split("\n\n")[0] for c in chunks] == ["Policy", "Travel", "Security"]
    check_offsets(chunks, pages)


def test_heading_without_body_stays_with_the_next_section() -> None:
    pages = [page(1, (1, "Handbook"), (2, "Leave"), "Leave body.")]
    (chunk,) = chunk_pages(pages, ChunkingConfig())
    assert chunk.text == "Handbook\n\nLeave\n\nLeave body."
    assert chunk.heading_path == ("Handbook", "Leave")


def test_long_sections_are_packed_with_overlap() -> None:
    paragraphs = [paragraph(i) for i in range(12)]  # ~50 tokens each
    pages = [page(1, (1, "Rules"), *paragraphs)]
    config = ChunkingConfig(target_tokens=200, overlap_ratio=0.3)
    chunks = chunk_pages(pages, config)

    assert len(chunks) > 2
    assert all(c.token_count <= 200 * 1.3 for c in chunks)
    assert all(c.heading_path == ("Rules",) for c in chunks)
    for prev, nxt in pairwise(chunks):
        assert nxt.text.startswith(prev.text.split("\n\n")[-1])  # one paragraph of overlap
    assert all(any(p in c.text for c in chunks) for p in paragraphs)
    check_offsets(chunks, pages)


def test_no_overlap_across_sections() -> None:
    pages = [page(1, (1, "A"), paragraph(1), (1, "B"), paragraph(2))]
    chunks = chunk_pages(pages, ChunkingConfig(target_tokens=60, overlap_ratio=0.9))
    assert [c.text.split("\n\n")[0] for c in chunks] == ["A", "B"]


def test_oversized_paragraph_split_by_sentence() -> None:
    sentences = [f"Sentence {i} says something about policy number {i}." for i in range(80)]
    pages = [page(1, " ".join(sentences))]
    chunks = chunk_pages(pages, ChunkingConfig(target_tokens=100, overlap_ratio=0.15))
    assert len(chunks) > 3
    assert all(c.token_count <= 115 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)
    check_offsets(chunks, pages)


def test_unbreakable_text_is_hard_split() -> None:
    blob = "x" * 5000
    chunks = chunk_pages([page(1, blob)], ChunkingConfig(target_tokens=100, overlap_ratio=0))
    assert "".join(c.text for c in chunks) == blob
    assert all(c.token_count <= 100 for c in chunks)


def test_chunks_span_pages_within_a_section() -> None:
    pages = [page(1, (1, "Scope"), "Starts on page one."), page(2, "Continues on page two.")]
    (chunk,) = chunk_pages(pages, ChunkingConfig())
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    assert chunk.text.endswith("Starts on page one.\n\nContinues on page two.")
    check_offsets([chunk], pages)


def test_empty_pages_give_no_chunks() -> None:
    assert chunk_pages([PageContent(1, "", [])], ChunkingConfig()) == []


def test_embedding_input_contextual_header() -> None:
    (chunk,) = chunk_pages([page(1, (1, "Leave"), (2, "Sick"), "Ten days.")], ChunkingConfig())
    assert embedding_input(chunk, "Handbook", contextual_header=True).startswith(
        "Handbook > Leave > Sick\n\n"
    )
    assert embedding_input(chunk, "Handbook", contextual_header=False) == chunk.text
