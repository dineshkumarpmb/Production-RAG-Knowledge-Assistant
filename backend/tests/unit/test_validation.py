import zipfile
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import FileType
from app.ingestion.validation import check_content, file_type_from_name, sanitize_filename
from tests import files


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\Q3 plan (v2).docx", "Q3 plan (v2).docx"),
        ("bad<>:name?.md", "bad___name_.md"),
        ("ctrl\x00\x1fchars.txt", "ctrlchars.txt"),
        ("", "upload"),
        ("...", "upload"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_long_filenames_keep_extension() -> None:
    name = sanitize_filename("a" * 300 + ".pdf")
    assert len(name) == 200
    assert name.endswith(".pdf")


def test_file_type_from_name() -> None:
    assert file_type_from_name("x.PDF") == FileType.PDF
    assert file_type_from_name("x.markdown") == FileType.MD
    with pytest.raises(AppError) as err:
        file_type_from_name("x.exe")
    assert err.value.status_code == 415


def expect_rejected(path: Path, file_type: FileType, settings: Settings, code: str) -> None:
    with pytest.raises(AppError) as err:
        check_content(path, file_type, settings)
    assert err.value.code == code


def test_valid_files_pass(tmp_path: Path, settings: Settings) -> None:
    check_content(files.handbook_pdf(tmp_path / "a.pdf"), FileType.PDF, settings)
    check_content(files.sample_docx(tmp_path / "a.docx"), FileType.DOCX, settings)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    check_content(tmp_path / "a.txt", FileType.TXT, settings)


def test_renamed_extensions_rejected(tmp_path: Path, settings: Settings) -> None:
    docx_path = files.sample_docx(tmp_path / "a.docx")
    pdf_path = files.handbook_pdf(tmp_path / "a.pdf")
    expect_rejected(docx_path, FileType.PDF, settings, "file_content_mismatch")
    expect_rejected(pdf_path, FileType.DOCX, settings, "file_content_mismatch")
    expect_rejected(pdf_path, FileType.TXT, settings, "file_content_mismatch")
    exe = tmp_path / "evil.txt"
    exe.write_bytes(b"MZ\x90\x00\x03")
    expect_rejected(exe, FileType.TXT, settings, "file_content_mismatch")


def test_plain_zip_is_not_docx(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "a.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("readme.txt", "hi")
    expect_rejected(path, FileType.DOCX, settings, "file_content_mismatch")


def test_macro_docx_rejected(tmp_path: Path, settings: Settings) -> None:
    path = files.sample_docx(tmp_path / "a.docx")
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("word/vbaProject.bin", b"\x00" * 10)
    expect_rejected(path, FileType.DOCX, settings, "macros_not_allowed")


def test_zip_bomb_rejected(tmp_path: Path, settings: Settings) -> None:
    settings = settings.model_copy(update={"max_docx_uncompressed_bytes": 1_000_000})
    path = files.sample_docx(tmp_path / "a.docx")
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/media/huge.bin", b"\x00" * 1_000_001)  # compresses to ~1 KB
    expect_rejected(path, FileType.DOCX, settings, "file_too_large")
