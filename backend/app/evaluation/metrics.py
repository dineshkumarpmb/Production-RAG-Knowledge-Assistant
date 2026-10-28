"""Retrieval metrics against gold references (document + page, optionally a section heading).

A retrieved chunk counts as relevant when it covers a gold reference that no earlier chunk in
the ranking has covered yet, so several chunks from the same gold page are not rewarded twice.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldRef:
    filename: str
    page: int
    heading: str | None = None  # substring of one heading in the chunk's heading path


@dataclass(frozen=True)
class RetrievedChunk:
    filename: str
    page_start: int
    page_end: int
    heading_path: tuple[str, ...] = ()


def covers(chunk: RetrievedChunk, gold: GoldRef) -> bool:
    return (
        chunk.filename == gold.filename
        and chunk.page_start <= gold.page <= chunk.page_end
        and (
            gold.heading is None
            or any(gold.heading.lower() in h.lower() for h in chunk.heading_path)
        )
    )


def gains(ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef]) -> list[int]:
    """1 at each rank whose chunk covers a not-yet-covered gold reference, else 0."""
    found: set[GoldRef] = set()
    result = []
    for chunk in ranking:
        new = {g for g in gold if g not in found and covers(chunk, g)}
        found |= new
        result.append(1 if new else 0)
    return result


def recall_at_k(ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef], k: int) -> float:
    found = {g for chunk in ranking[:k] for g in gold if covers(chunk, g)}
    return len(found) / len(gold)


def hit_at_k(ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef], k: int) -> float:
    return 1.0 if any(gains(ranking[:k], gold)) else 0.0


def reciprocal_rank(ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef]) -> float:
    for rank, gain in enumerate(gains(ranking, gold), start=1):
        if gain:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef], k: int) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains(ranking[:k], gold)))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(pct) - 1]
