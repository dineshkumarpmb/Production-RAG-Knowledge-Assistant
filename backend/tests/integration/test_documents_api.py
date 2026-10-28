import uuid
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import DocumentPage, Role, User
from tests import files
from tests.conftest import Identity, InlineQueue

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


async def upload(
    client: httpx.AsyncClient, path: Path, name: str | None = None, mime: str = "application/pdf"
) -> httpx.Response:
    return await client.post(
        "/documents", files={"file": (name or path.name, path.read_bytes(), mime)}
    )


def assert_error(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body: dict[str, Any] = response.json()
    assert body["code"] == code
    assert body["request_id"] == response.headers["x-request-id"]
    return body


async def test_pdf_upload_is_parsed_with_metadata(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    response = await upload(client, files.handbook_pdf(tmp_path / "handbook.pdf", pages=3))
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["duplicate"] is False
    doc_id = body["document"]["id"]

    doc = (await client.get(f"/documents/{doc_id}")).json()
    assert doc["status"] == "ready"
    assert doc["error"] is None
    assert doc["page_count"] == 3
    assert doc["title"] == "ACME Employee Handbook"
    assert doc["author"] == "HR Team"
    assert doc["language"] == "en"
    assert doc["file_type"] == "pdf"
    assert doc["created_at_src"].startswith("2024-01-15")
    assert doc["metadata"]["page_mode"] == "physical"
    assert doc["metadata"]["removed_header_footer_blocks"] == 6
    assert doc["metadata"]["needs_ocr"] is False

    pages = (await client.get(f"/documents/{doc_id}/pages")).json()
    assert pages["total"] == 3
    assert [p["page_number"] for p in pages["items"]] == [1, 2, 3]
    first = pages["items"][0]
    assert first["headings"] == [{"text": "Section 1: Policy Area 1", "level": 1, "char_offset": 0}]
    assert "ACME Corp Employee Handbook" not in first["text"]


async def test_docx_and_markdown_uploads(client: httpx.AsyncClient, tmp_path: Path) -> None:
    docx_resp = await upload(client, files.sample_docx(tmp_path / "travel.docx"), mime=DOCX_MIME)
    doc = (await client.get(f"/documents/{docx_resp.json()['document']['id']}")).json()
    assert (doc["status"], doc["page_count"], doc["title"]) == ("ready", 2, "Travel Policy")
    assert doc["mime_type"] == DOCX_MIME

    md = tmp_path / "guide.md"
    md.write_text("# Setup Guide\n\nInstall the package and run the tests.\n", encoding="utf-8")
    md_resp = await upload(client, md, mime="text/markdown")
    doc = (await client.get(f"/documents/{md_resp.json()['document']['id']}")).json()
    assert (doc["status"], doc["page_count"], doc["title"]) == ("ready", 1, "Setup Guide")


async def test_exact_duplicate_returns_existing_document(
    client: httpx.AsyncClient, tmp_path: Path, queue: InlineQueue
) -> None:
    path = files.handbook_pdf(tmp_path / "h.pdf")
    first = await upload(client, path)
    second = await upload(client, path, name="copy-of-handbook.pdf")
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document"]["id"] == first.json()["document"]["id"]
    assert len(queue.enqueued) == 1
    assert (await client.get("/documents")).json()["total"] == 1


async def test_rejected_uploads(client: httpx.AsyncClient, tmp_path: Path) -> None:
    docx_path = files.sample_docx(tmp_path / "real.docx")
    assert_error(await upload(client, docx_path, name="renamed.pdf"), 415, "file_content_mismatch")

    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ\x90\x00")
    assert_error(await upload(client, exe), 415, "unsupported_file_type")

    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    assert_error(await upload(client, empty, mime="text/plain"), 422, "empty_file")

    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (2 * 1024 * 1024))  # limit is 1 MB in tests
    assert_error(await upload(client, big, mime="text/plain"), 413, "file_too_large")

    assert (await client.get("/documents")).json()["total"] == 0


async def test_corrupt_and_empty_pdfs_fail_with_reason(
    client: httpx.AsyncClient, tmp_path: Path
) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.7\nnot actually a pdf at all")
    doc_id = (await upload(client, corrupt)).json()["document"]["id"]
    doc = (await client.get(f"/documents/{doc_id}")).json()
    assert doc["status"] == "failed"
    assert "could not be opened" in doc["error"]

    scanned = files.empty_pdf(tmp_path / "scan.pdf", pages=3)
    doc_id = (await upload(client, scanned)).json()["document"]["id"]
    doc = (await client.get(f"/documents/{doc_id}")).json()
    assert doc["status"] == "failed"
    assert "OCR" in doc["error"]


async def test_reprocess_is_idempotent(
    client: httpx.AsyncClient, tmp_path: Path, sync_sessions: sessionmaker[Session]
) -> None:
    doc_id = (await upload(client, files.handbook_pdf(tmp_path / "h.pdf", pages=3))).json()[
        "document"
    ]["id"]
    before = (await client.get(f"/documents/{doc_id}/pages")).json()["items"]

    response = await client.post(f"/documents/{doc_id}/reprocess")
    assert response.status_code == 202
    assert response.json()["status"] == "ready"  # inline queue already finished it

    after = (await client.get(f"/documents/{doc_id}/pages")).json()["items"]
    assert after == before
    with sync_sessions() as s:
        count = s.scalar(
            select(func.count())
            .select_from(DocumentPage)
            .where(DocumentPage.document_id == uuid.UUID(doc_id))
        )
    assert count == 3


async def test_queue_outage_marks_document_failed(
    client: httpx.AsyncClient, tmp_path: Path, queue: InlineQueue
) -> None:
    queue.fail = True
    response = await upload(client, files.handbook_pdf(tmp_path / "h.pdf"))
    assert_error(response, 503, "queue_unavailable")

    (doc,) = (await client.get("/documents")).json()["items"]
    assert doc["status"] == "failed"

    queue.fail = False
    assert (await client.post(f"/documents/{doc['id']}/reprocess")).json()["status"] == "ready"


async def test_private_documents_hidden_from_other_users(
    client: httpx.AsyncClient,
    tmp_path: Path,
    identity: Identity,
    sync_sessions: sessionmaker[Session],
    tenant_and_user: tuple[uuid.UUID, uuid.UUID],
) -> None:
    doc_id = (await upload(client, files.handbook_pdf(tmp_path / "h.pdf"))).json()["document"]["id"]

    with sync_sessions() as s, s.begin():
        bob = User(tenant_id=tenant_and_user[0], email="bob@acme.test", role=Role.EDITOR)
        s.add(bob)
        s.flush()
        identity.user_id = bob.id

    assert (await client.get("/documents")).json()["total"] == 0
    assert_error(await client.get(f"/documents/{doc_id}"), 404, "not_found")
    assert_error(await client.get(f"/documents/{doc_id}/pages"), 404, "not_found")
    assert_error(await client.post(f"/documents/{doc_id}/reprocess"), 404, "not_found")
    # Re-uploading the same bytes must not reveal the private document's details.
    body = assert_error(
        await upload(client, tmp_path / "h.pdf", name="mine.pdf"), 409, "duplicate_document"
    )
    assert doc_id not in body["message"]


async def test_error_schema_and_request_id(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/documents/{uuid.uuid4()}", headers={"X-Request-ID": "abc-123"})
    assert assert_error(response, 404, "not_found")["request_id"] == "abc-123"

    assert_error(await client.get("/documents/not-a-uuid"), 422, "validation_error")
    assert (await client.get("/healthz")).json() == {"status": "ok"}
