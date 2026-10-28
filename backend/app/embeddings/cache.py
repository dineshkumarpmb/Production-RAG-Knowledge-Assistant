"""Embedding with a database cache keyed by (hash of the exact input text, model)."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import EmbeddingCache
from app.embeddings.base import EmbeddingProvider
from app.ingestion.dedup import sha256_text

_LOOKUP_BATCH = 500
_EMBED_BATCH = 256  # inputs per provider call; each batch is cached before the next starts


@dataclass
class EmbeddingStats:
    model: str
    inputs: int = 0
    unique_inputs: int = 0
    cache_hits: int = 0
    embedded: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lookup(session: Session, model: str, hashes: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for i in range(0, len(hashes), _LOOKUP_BATCH):
        rows = session.execute(
            select(EmbeddingCache.text_hash, EmbeddingCache.embedding).where(
                EmbeddingCache.model == model,
                EmbeddingCache.text_hash.in_(hashes[i : i + _LOOKUP_BATCH]),
            )
        )
        found.update({h: [float(x) for x in v] for h, v in rows})
    return found


def _store(session: Session, model: str, items: dict[str, list[float]]) -> None:
    insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
    rows = [{"text_hash": h, "model": model, "embedding": v} for h, v in items.items()]
    session.execute(insert(EmbeddingCache).values(rows).on_conflict_do_nothing())


def embed_with_cache(
    session_factory: sessionmaker[Session], provider: EmbeddingProvider, texts: Sequence[str]
) -> tuple[list[list[float]], EmbeddingStats]:
    """Embed `texts` (in order), only paying for texts this model has never embedded before."""
    stats = EmbeddingStats(model=provider.model_name, inputs=len(texts))
    hashes = [sha256_text(t) for t in texts]
    unique = dict(zip(hashes, texts, strict=True))  # dedupes identical inputs, keeps order
    stats.unique_inputs = len(unique)

    with session_factory() as session:
        vectors = _lookup(session, provider.model_name, list(unique))
    stats.cache_hits = len(vectors)

    missing = [h for h in unique if h not in vectors]
    for i in range(0, len(missing), _EMBED_BATCH):
        batch = missing[i : i + _EMBED_BATCH]
        result = provider.embed_documents([unique[h] for h in batch])
        new = dict(zip(batch, result.vectors, strict=True))
        with session_factory() as session, session.begin():
            _store(session, provider.model_name, new)
        vectors.update(new)
        stats.embedded += len(batch)
        stats.tokens += result.tokens

    stats.cost_usd = round(stats.tokens * provider.price_per_million_tokens / 1_000_000, 6)
    return [vectors[h] for h in hashes], stats
