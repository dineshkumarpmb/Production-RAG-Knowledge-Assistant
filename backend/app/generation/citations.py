"""Map Claude's citations back to our documents, pages and chunks.

Each context passage is sent as document `i` (0-based) whose content blocks are the passage's
paragraphs (`ContextBlock.segments`). A `content_block_location` citation names the document
and a block range (end exclusive), which identifies the exact paragraphs and so their chunks
and pages.
"""

from collections.abc import Sequence
from typing import Any

from app.generation.types import Citation
from app.retrieval.context import ContextBlock


class CitationResolver:
    def __init__(self, blocks: Sequence[ContextBlock]) -> None:
        self._blocks = list(blocks)

    def resolve(self, citation: Any) -> Citation | None:
        """None for a citation that points outside the documents we sent (never expected)."""
        index = getattr(citation, "document_index", None)
        if index is None or not 0 <= index < len(self._blocks):
            return None
        block = self._blocks[index]
        start = getattr(citation, "start_block_index", None)
        end = getattr(citation, "end_block_index", None)
        segments = block.segments[start:end] if start is not None and end is not None else []
        if not segments:  # unexpected location type: cite the whole passage
            segments = block.segments

        chunk_ids = list(dict.fromkeys(s.chunk_id for s in segments)) or block.chunk_ids
        return Citation(
            source=index + 1,
            document_id=block.document_id,
            document_title=block.document_title,
            filename=block.filename,
            page_start=min((s.page_start for s in segments), default=block.page_start),
            page_end=max((s.page_end for s in segments), default=block.page_end),
            heading_path=block.heading_path,
            chunk_ids=chunk_ids,
            cited_text=getattr(citation, "cited_text", "") or "",
        )
