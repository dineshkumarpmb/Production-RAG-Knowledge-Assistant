"""Database objects created by migrations but deliberately not mapped in the ORM models."""

from typing import Any

# chunks.tsv is a generated full-text column used by keyword search (migration 0003).
UNMAPPED = {("column", "tsv"), ("index", "ix_chunks_tsv")}


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Alembic hook: ignore UNMAPPED objects when comparing the database with the models."""
    return not (reflected and compare_to is None and (type_, name) in UNMAPPED)
