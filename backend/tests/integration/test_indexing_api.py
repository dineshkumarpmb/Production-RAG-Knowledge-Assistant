import random
import uuid
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Role, User
from app.ingestion.pipeline import index_document, stale_document_ids
from app.storage.local import LocalStorage
from tests import files
from tests.conftest import FakeEmbeddingProvider, Identity


async def upload(client: httpx.AsyncClient, path: Path, mime: str = "text/plain") -> dict[str, Any]:
    response = await client.post("/documents", files={"file": (path.name, path.read_bytes(), mime)})
    assert response.status_code in (200, 202), response.text
    doc: dict[str, Any] = response.json()["document"]
    return (await client.get(f"/documents/{doc['id']}")).json()  # type: ignore[no-any-return]


def essay_file(path: Path, *, seed: int = 7, change_word: int | None = None) -> Path:
    rng = random.Random(seed)
    vocab = [f"term{i}" for i in range(300)]
    words = [rng.choice(vocab) for _ in range(600)]
    if change_word is not None:
        words[change_word] = "revised"
    sentences = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, 600, 12)]
    paragraphs = [" ".join(sentences[i : i + 5]) for i in range(0, len(sentences), 5)]
    path.write_text("# Retention Policy\n\n" + "\n\n".join(paragraphs), encoding="utf-8")
    return path


def add_user(
    sessions: sessionmaker[Session], tenant_id: uuid.UUID, email: str, role: Role
) -> uuid.UUID:
    with sessions() as s, s.begin():
        user = User(tenant_id=tenant_id, email=email, role=role)
        s.add(user)
        s.flush()
        return user.id


async def test_upload_is_chunked_and_embedded(
    client: httpx.AsyncClient,
    tmp_path: Path,
    embedder: FakeEmbeddingProvider,
    sync_sessions: sessionmaker[Session],
) -> None:
    doc = await upload(client, files.handbook_pdf(tmp_path / "h.pdf", pages=3), "application/pdf")
    assert doc["status"] == "ready"
    assert doc["metadata"]["chunking"]["chunk_count"] == 3  # one section per page
    assert doc["metadata"]["embedding"]["model"] == "fake-embed-1"
    assert doc["metadata"]["embedding"]["embedded"] == 3

    chunks = (await client.get(f"/documents/{doc['id']}/chunks")).json()
    assert chunks["total"] == 3
    for i, chunk in enumerate(chunks["items"], start=1):
        assert (chunk["page_start"], chunk["page_end"]) == (i, i)
        assert chunk["heading_path"] == [f"Section {i}: Policy Area {i}"]
        assert chunk["embedding_model"] == "fake-embed-1"
        assert f"(page {i})" in chunk["text"]

    # The contextual header is embedded but not stored as chunk text.
    assert embedder.embedded[0].startswith("ACME Employee Handbook > Section 1: Policy Area 1\n\n")
    with sync_sessions() as s:
        stored = s.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert len(stored[0].embedding) == 1024


async def test_reprocessing_hits_the_embedding_cache(
    client: httpx.AsyncClient, tmp_path: Path, embedder: FakeEmbeddingProvider
) -> None:
    doc = await upload(client, essay_file(tmp_path / "policy.md"), "text/markdown")
    first_calls = len(embedder.embedded)
    assert first_calls == doc["metadata"]["chunking"]["chunk_count"] > 1

    response = await client.post(f"/documents/{doc['id']}/reprocess")
    stats = response.json()["metadata"]["embedding"]
    assert len(embedder.embedded) == first_calls
    assert (stats["embedded"], stats["tokens"], stats["cost_usd"]) == (0, 0, 0)
    assert stats["cache_hits"] == stats["unique_inputs"]


async def test_near_duplicate_flagged_then_kept(client: httpx.AsyncClient, tmp_path: Path) -> None:
    original = await upload(client, essay_file(tmp_path / "v1.md"), "text/markdown")
    revised = await upload(client, essay_file(tmp_path / "v2.md", change_word=300), "text/markdown")

    assert original["near_duplicate_of_id"] is None
    assert revised["status"] == "ready"
    assert revised["near_duplicate_of_id"] == original["id"]
    assert revised["metadata"]["near_duplicate"]["distance"] <= 3

    kept = (
        await client.post(
            f"/documents/{revised['id']}/near-duplicate", json={"action": "keep_both"}
        )
    ).json()
    assert kept["near_duplicate_of_id"] is None
    assert "near_duplicate" not in kept["metadata"]

    # The decision sticks when the document is processed again.
    again = (await client.post(f"/documents/{revised['id']}/reprocess")).json()
    assert again["near_duplicate_of_id"] is None


async def test_near_duplicate_replace_deletes_old_version(
    client: httpx.AsyncClient, tmp_path: Path, storage: LocalStorage
) -> None:
    original = await upload(client, essay_file(tmp_path / "v1.md"), "text/markdown")
    revised = await upload(client, essay_file(tmp_path / "v2.md", change_word=10), "text/markdown")

    response = await client.post(
        f"/documents/{revised['id']}/near-duplicate", json={"action": "replace"}
    )
    assert response.status_code == 200, response.text
    replaced = response.json()
    assert replaced["version"] == 2
    assert replaced["near_duplicate_of_id"] is None
    assert replaced["metadata"]["replaced_document"]["document_id"] == original["id"]

    assert (await client.get(f"/documents/{original['id']}")).status_code == 404
    (remaining,) = storage.root.rglob("original.*")  # only the new version's file is left
    assert revised["id"] in remaining.as_posix()

    response = await client.post(
        f"/documents/{revised['id']}/near-duplicate", json={"action": "keep_both"}
    )
    assert response.status_code == 409


async def test_near_duplicate_never_points_at_invisible_documents(
    client: httpx.AsyncClient,
    tmp_path: Path,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    await upload(client, essay_file(tmp_path / "alice.md"), "text/markdown")
    identity.user_id = add_user(sync_sessions, tenant_and_user[0], "bob@acme.test", Role.EDITOR)
    bobs = await upload(client, essay_file(tmp_path / "bob.md", change_word=5), "text/markdown")
    assert bobs["near_duplicate_of_id"] is None


async def test_embedding_failure_marks_document_failed(
    client: httpx.AsyncClient, tmp_path: Path, embedder: FakeEmbeddingProvider
) -> None:
    embedder.fail = True
    doc = await upload(client, essay_file(tmp_path / "p.md"), "text/markdown")
    assert doc["status"] == "failed"
    assert doc["error"] == "Embedding failed: fake provider is down"
    assert doc["page_count"] == 1  # parsing succeeded; only embedding failed

    embedder.fail = False
    assert (await client.post(f"/documents/{doc['id']}/reprocess")).json()["status"] == "ready"


async def test_delete_removes_everything(
    client: httpx.AsyncClient,
    tmp_path: Path,
    storage: LocalStorage,
    sync_sessions: sessionmaker[Session],
) -> None:
    doc = await upload(client, essay_file(tmp_path / "p.md"), "text/markdown")
    assert (await client.delete(f"/documents/{doc['id']}")).status_code == 204

    assert (await client.get(f"/documents/{doc['id']}")).status_code == 404
    assert (await client.get("/documents")).json()["total"] == 0
    assert list(storage.root.rglob("original.*")) == []
    with sync_sessions() as s:
        assert s.scalars(select(Chunk)).all() == []


async def test_ingestion_report_for_admins(
    client: httpx.AsyncClient,
    tmp_path: Path,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    assert (await client.get("/admin/ingestion-report")).status_code == 403  # editor

    identity.user_id = add_user(sync_sessions, tenant_and_user[0], "root@acme.test", Role.ADMIN)
    pdf = files.handbook_pdf(tmp_path / "h.pdf", pages=3)
    await upload(client, pdf, "application/pdf")
    await upload(client, pdf, "application/pdf")  # exact duplicate
    await upload(client, essay_file(tmp_path / "p.md"), "text/markdown")

    report = (await client.get("/admin/ingestion-report")).json()
    assert report["documents_total"] == 2
    assert report["documents_by_status"] == {"ready": 2}
    assert report["pages_total"] == 4
    assert report["chunks_total"] > 3
    assert report["chunks_by_embedding_model"] == {"fake-embed-1": report["chunks_total"]}
    assert report["exact_duplicate_uploads"] == 1
    assert report["near_duplicates_pending"] == 0
    assert report["duplicate_chunks"] == 0
    assert report["embedding_tokens"] > 0
    assert report["embedding_cost_usd"] > 0


async def test_reembedding_after_model_change(
    client: httpx.AsyncClient,
    tmp_path: Path,
    settings: Settings,
    sync_sessions: sessionmaker[Session],
) -> None:
    doc = await upload(client, essay_file(tmp_path / "p.md"), "text/markdown")
    doc_id = uuid.UUID(doc["id"])
    assert stale_document_ids(sync_sessions, "fake-embed-1") == []
    assert stale_document_ids(sync_sessions, "fake-embed-2") == [doc_id]

    new_model = FakeEmbeddingProvider(model_name="fake-embed-2")
    index_document(
        doc_id, session_factory=sync_sessions, settings=settings, embedder_factory=lambda: new_model
    )
    assert stale_document_ids(sync_sessions, "fake-embed-2") == []
    chunks = (await client.get(f"/documents/{doc_id}/chunks")).json()["items"]
    assert {c["embedding_model"] for c in chunks} == {"fake-embed-2"}
