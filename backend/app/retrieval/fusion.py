"""Reciprocal Rank Fusion: score(d) = sum over retrievers of weight / (k + rank(d)).

RRF only uses ranks, so the keyword scores (ts_rank_cd) and vector scores (cosine similarity)
never need to be put on a common scale.
"""

from collections.abc import Mapping, Sequence

from app.retrieval.types import FusedChunk, RankedChunk, Retriever


def reciprocal_rank_fusion(
    rankings: Mapping[Retriever, Sequence[RankedChunk]],
    *,
    k: int = 60,
    weights: Mapping[Retriever, float] | None = None,
) -> list[FusedChunk]:
    fused: dict[object, FusedChunk] = {}
    for retriever, ranking in rankings.items():
        weight = (weights or {}).get(retriever, 1.0)
        rank = 0
        for hit in ranking:
            entry = fused.setdefault(hit.chunk_id, FusedChunk(hit.chunk_id, 0.0))
            if retriever in entry.ranks:
                continue  # ignore repeats within one ranking
            rank += 1
            entry.ranks[retriever] = rank
            entry.scores[retriever] = hit.score
            entry.score += weight / (k + rank)
    # Ties: the better single-retriever rank wins, then chunk id for determinism.
    return sorted(
        fused.values(),
        key=lambda f: (-f.score, min(f.ranks.values(), default=0), str(f.chunk_id)),
    )
