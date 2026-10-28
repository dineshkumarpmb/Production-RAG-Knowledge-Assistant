"""Duplicate detection.

- Exact documents: SHA-256 of the uploaded bytes, unique per tenant (checked at upload).
- Near-duplicate documents: 64-bit SimHash over word 3-grams; a small Hamming distance means
  the texts are almost the same (typically a new version of a document).
- Chunks: hash of normalised chunk text, so repeated boilerplate can be spotted and collapsed.
"""

import hashlib
import re
import uuid
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.access import Principal, can_read
from app.db.models import Document, DocumentStatus

_WORD = re.compile(r"\w+")
_WHITESPACE = re.compile(r"\s+")
_SHINGLE = 3
_BITS = 64
_MASK = (1 << _BITS) - 1
_SIGN_BIT = 1 << (_BITS - 1)


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def sha256_text(text: str) -> str:
    """Hash of the exact text (no normalisation), for caching what was sent to a model."""
    return hashlib.sha256(text.encode()).hexdigest()


def simhash(text: str) -> int:
    """Charikar SimHash of word 3-grams, weighted by frequency. Returns an unsigned 64-bit int."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0
    if len(words) < _SHINGLE:
        features = Counter([" ".join(words)])
    else:
        features = Counter(
            " ".join(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)
        )

    hashes = np.array(
        [
            int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            for f in features
        ],
        dtype=">u8",
    )
    weights = np.array(list(features.values()), dtype=np.int64)
    bits = np.unpackbits(hashes.view(np.uint8)).reshape(-1, _BITS).astype(np.int64)
    score = ((bits * 2 - 1) * weights[:, None]).sum(axis=0)  # +w where the bit is 1, -w where 0
    return int("".join("1" if s > 0 else "0" for s in score), 2)


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & _MASK).bit_count()


def to_signed64(value: int) -> int:
    """Postgres BIGINT is signed; store the unsigned hash in two's complement."""
    return value - (1 << _BITS) if value >= _SIGN_BIT else value


def from_signed64(value: int) -> int:
    return value + (1 << _BITS) if value < 0 else value


@dataclass(frozen=True)
class NearDuplicate:
    document_id: uuid.UUID
    title: str | None
    distance: int


def find_near_duplicate(
    session: Session,
    doc: Document,
    fingerprint: int,
    max_distance: int,
    *,
    exclude: Collection[uuid.UUID] = (),
) -> NearDuplicate | None:
    """Closest ready document within `max_distance` that the uploader can see.

    Compares in Python against every candidate's fingerprint, which is fine for thousands of
    documents per tenant; beyond that, switch to LSH bands (split the hash into 4 x 16 bits).
    """
    rows = session.execute(
        select(Document.id, Document.title, Document.simhash).where(
            Document.tenant_id == doc.tenant_id,
            Document.id != doc.id,
            Document.simhash.is_not(None),
            Document.status == DocumentStatus.READY,
            # Only documents the uploader may read, so the flag never leaks private documents.
            can_read(Principal(user_id=doc.owner_id, tenant_id=doc.tenant_id)),
        )
    ).all()
    best: NearDuplicate | None = None
    for doc_id, title, stored in rows:
        if doc_id in exclude or stored is None:
            continue
        distance = hamming(fingerprint, from_signed64(stored))
        if distance <= max_distance and (best is None or distance < best.distance):
            best = NearDuplicate(doc_id, title, distance)
    return best
