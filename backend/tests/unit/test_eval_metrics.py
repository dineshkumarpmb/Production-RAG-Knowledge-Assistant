import math
from pathlib import Path

import pytest

from app.evaluation.metrics import (
    GoldRef,
    RetrievedChunk,
    covers,
    gains,
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.retrieval import load_dataset

GOLD_A = GoldRef("a.pdf", 3)
GOLD_B = GoldRef("b.md", 1, heading="Hotels")


def chunk(filename: str, start: int, end: int | None = None, *path: str) -> RetrievedChunk:
    return RetrievedChunk(filename, start, end or start, path)


def test_covers_checks_file_page_range_and_heading() -> None:
    assert covers(chunk("a.pdf", 2, 4), GOLD_A)
    assert not covers(chunk("a.pdf", 4, 5), GOLD_A)
    assert not covers(chunk("x.pdf", 3), GOLD_A)
    assert covers(chunk("b.md", 1, 1, "Travel Policy", "Hotels"), GOLD_B)
    assert covers(chunk("b.md", 1, 1, "Travel Policy", "hotels and meals"), GOLD_B)
    assert not covers(chunk("b.md", 1, 1, "Travel Policy", "Meals"), GOLD_B)


def test_each_gold_reference_is_rewarded_once() -> None:
    ranking = [
        chunk("x.pdf", 1),
        chunk("a.pdf", 3),
        chunk("a.pdf", 3),
        chunk("b.md", 1, 1, "Hotels"),
    ]
    assert gains(ranking, [GOLD_A, GOLD_B]) == [0, 1, 0, 1]


def test_ranking_metrics() -> None:
    ranking = [
        chunk("x.pdf", 1),
        chunk("a.pdf", 3),
        chunk("y.pdf", 1),
        chunk("b.md", 1, 1, "Hotels"),
    ]
    gold = [GOLD_A, GOLD_B]
    assert recall_at_k(ranking, gold, 1) == 0
    assert recall_at_k(ranking, gold, 2) == 0.5
    assert recall_at_k(ranking, gold, 4) == 1
    assert hit_at_k(ranking, gold, 1) == 0
    assert hit_at_k(ranking, gold, 2) == 1
    assert reciprocal_rank(ranking, gold) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, gold, 10) == pytest.approx(expected)
    assert ndcg_at_k([chunk("a.pdf", 3)], [GOLD_A], 10) == 1
    assert reciprocal_rank([chunk("x.pdf", 1)], [GOLD_A]) == 0


def test_percentile() -> None:
    assert percentile([], 95) == 0
    assert percentile([7.0], 95) == 7
    assert percentile([float(i) for i in range(1, 101)], 50) == pytest.approx(50.5)


def test_load_dataset(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"id": "q1", "question": "Hotel limit?", "type": "single-fact",'
        ' "gold": [{"filename": "b.md", "page": 1, "heading": "Hotels"}]}\n'
        "\n"
        '{"id": "q2", "question": "Pets?", "gold": []}\n',
        encoding="utf-8",
    )
    q1, q2 = load_dataset(path)
    assert q1.gold == (GOLD_B,)
    assert (q2.gold, q2.type) == ((), "unspecified")

    path.write_text(
        '{"id": "q1", "question": "a"}\n{"id": "q1", "question": "b"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)
    path.write_text('{"question": "no id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"qa\.jsonl:1"):
        load_dataset(path)


def test_sample_dataset_is_valid() -> None:
    root = Path(__file__).resolve().parents[3] / "eval"
    questions = load_dataset(root / "datasets" / "sample_qa.jsonl")
    corpus = {p.name for p in (root / "corpus" / "sample").iterdir()}
    assert len(questions) == 14
    assert {g.filename for q in questions for g in q.gold} <= corpus
