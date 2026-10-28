import random

import pytest

from app.ingestion.dedup import from_signed64, hamming, simhash, text_hash, to_signed64


def essay(seed: int, words: int = 400) -> list[str]:
    rng = random.Random(seed)
    vocab = [f"word{i}" for i in range(500)]
    return [rng.choice(vocab) for _ in range(words)]


def test_near_identical_texts_have_close_simhashes() -> None:
    original = essay(1)
    edited = original.copy()
    edited[200] = "changed"
    assert hamming(simhash(" ".join(original)), simhash(" ".join(edited))) <= 3
    # Case and whitespace do not matter.
    assert simhash(" ".join(original)) == simhash("  ".join(w.upper() for w in original))


def test_different_texts_are_far_apart() -> None:
    assert hamming(simhash(" ".join(essay(1))), simhash(" ".join(essay(2)))) > 10


def test_short_and_empty_texts() -> None:
    assert simhash("") == 0
    assert simhash("hi there") == simhash("Hi  there")


@pytest.mark.parametrize("value", [0, 1, 2**63 - 1, 2**63, 2**64 - 1])
def test_signed_round_trip(value: int) -> None:
    signed = to_signed64(value)
    assert -(2**63) <= signed < 2**63
    assert from_signed64(signed) == value


def test_text_hash_ignores_case_and_whitespace() -> None:
    assert text_hash("All rights  reserved.\n") == text_hash("all rights reserved.")
    assert text_hash("a") != text_hash("b")
