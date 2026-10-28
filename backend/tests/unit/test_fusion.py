import uuid

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RankedChunk, Retriever

KW, VEC = Retriever.KEYWORD, Retriever.VECTOR
A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))


def ranked(*ids: uuid.UUID) -> list[RankedChunk]:
    return [RankedChunk(i, 1.0 / n) for n, i in enumerate(ids, 1)]


def test_rrf_scores_and_order() -> None:
    fused = reciprocal_rank_fusion({KW: ranked(A, B, C), VEC: ranked(B, D, A)}, k=60)
    by_id = {f.chunk_id: f for f in fused}

    assert by_id[B].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[A].score == pytest.approx(1 / 61 + 1 / 63)
    assert by_id[C].score == pytest.approx(1 / 63)
    assert by_id[D].ranks == {VEC: 2}
    assert [f.chunk_id for f in fused] == [B, A, D, C]
    assert by_id[A].scores == {KW: 1.0, VEC: pytest.approx(1 / 3)}


def test_found_by_both_beats_top_of_one() -> None:
    fused = reciprocal_rank_fusion({KW: ranked(A, C), VEC: ranked(B, C)})
    assert fused[0].chunk_id == C


def test_weights() -> None:
    fused = reciprocal_rank_fusion({KW: ranked(A), VEC: ranked(B)}, weights={KW: 1.0, VEC: 2.0})
    assert [f.chunk_id for f in fused] == [B, A]
    zero = reciprocal_rank_fusion({KW: ranked(A), VEC: ranked(B)}, weights={KW: 0.0})
    assert zero[0].chunk_id == B


def test_single_ranking_keeps_its_order() -> None:
    assert [f.chunk_id for f in reciprocal_rank_fusion({KW: ranked(C, A, B)})] == [C, A, B]


def test_repeats_within_a_ranking_are_ignored() -> None:
    fused = reciprocal_rank_fusion({KW: ranked(A, A, B)}, k=60)
    by_id = {f.chunk_id: f for f in fused}
    assert by_id[A].score == pytest.approx(1 / 61)
    assert by_id[B].ranks == {KW: 2}


def test_ties_break_on_best_rank_then_id() -> None:
    fused = reciprocal_rank_fusion({KW: ranked(B), VEC: ranked(A)})
    assert [f.chunk_id for f in fused] == [A, B]  # equal score and rank: lower id first


def test_empty() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({KW: []}) == []
