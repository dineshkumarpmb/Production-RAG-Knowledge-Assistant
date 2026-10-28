import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    title: str | None
    filename: str
    file_type: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: str
    error: str | None
    page_count: int | None
    author: str | None
    created_at_src: datetime | None
    language: str | None
    visibility: str
    version: int
    metadata: dict[str, Any] = Field(validation_alias="meta", serialization_alias="metadata")
    near_duplicate_of_id: uuid.UUID | None = Field(
        description="An existing document this one is probably a new version of. "
        "Resolve with POST /documents/{id}/near-duplicate."
    )
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document: DocumentOut
    duplicate: bool = Field(description="True if this exact file was already uploaded.")


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class HeadingOut(BaseModel):
    text: str
    level: int
    char_offset: int


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    text: str
    char_count: int
    is_low_text: bool
    headings: list[HeadingOut]


class PageList(BaseModel):
    document_id: uuid.UUID
    page_mode: str | None
    items: list[PageOut]
    total: int


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_index: int
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    heading_path: list[str]
    token_count: int
    text_hash: str
    embedding_model: str | None


class ChunkList(BaseModel):
    document_id: uuid.UUID
    items: list[ChunkOut]
    total: int


class NearDuplicateResolution(BaseModel):
    action: Literal["keep_both", "replace"] = Field(
        description="keep_both: keep both documents. "
        "replace: delete the older document; this one becomes its next version."
    )


class IngestionReport(BaseModel):
    documents_total: int
    documents_by_status: dict[str, int]
    pages_total: int
    chunks_total: int
    avg_chunk_tokens: float | None
    duplicate_chunks: int = Field(description="Chunks whose normalised text appears more than once")
    chunks_by_embedding_model: dict[str, int]
    exact_duplicate_uploads: int
    near_duplicates_pending: int
    embedding_tokens: int = Field(description="Tokens billed in each document's latest indexing")
    embedding_cost_usd: float
    embedding_cache_hits: int
