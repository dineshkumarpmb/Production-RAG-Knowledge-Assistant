"""Upload checks: the extension must be supported AND the bytes must match it."""

import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import FileType

EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
    ".md": FileType.MD,
    ".markdown": FileType.MD,
}

MIME_TYPES: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileType.TXT: "text/plain",
    FileType.MD: "text/markdown",
}

# Signatures that must never appear at the start of a "text" file.
_BINARY_SIGNATURES = (b"%PDF-", b"PK\x03\x04", b"MZ", b"\x7fELF", b"\xd0\xcf\x11\xe0")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ()\[\]]")
_MAX_FILENAME_LENGTH = 200


def sanitize_filename(raw: str) -> str:
    """Keep only the base name, drop control/path characters and cap the length."""
    name = PureWindowsPath(PurePosixPath(raw).name).name  # strips both / and \ directories
    name = unicodedata.normalize("NFC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: _MAX_FILENAME_LENGTH - len(ext) - 1] + dot + ext) if dot else name[:200]
    return name or "upload"


def file_type_from_name(filename: str) -> FileType:
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSIONS:
        allowed = ", ".join(sorted(EXTENSIONS))
        raise AppError(
            "unsupported_file_type", f"Unsupported file type '{ext}'. Allowed: {allowed}.", 415
        )
    return EXTENSIONS[ext]


def _mismatch(file_type: FileType) -> AppError:
    return AppError(
        "file_content_mismatch",
        f"The file content does not match its .{file_type.value} extension.",
        415,
    )


def _check_docx(path: Path, settings: Settings) -> None:
    if not zipfile.is_zipfile(path):
        raise _mismatch(FileType.DOCX)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise _mismatch(FileType.DOCX) from exc

    names = {info.filename for info in infos}
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise _mismatch(FileType.DOCX)
    if any(name.lower().endswith("vbaproject.bin") for name in names):
        raise AppError("macros_not_allowed", "Documents containing macros are not accepted.", 415)
    if sum(info.file_size for info in infos) > settings.max_docx_uncompressed_bytes:
        # Protects the parser from zip bombs.
        raise AppError("file_too_large", "The DOCX expands to more data than allowed.", 413)


def check_content(path: Path, file_type: FileType, settings: Settings) -> None:
    """Raise AppError unless the file's bytes are consistent with its declared type."""
    with path.open("rb") as f:
        head = f.read(8192)

    match file_type:
        case FileType.PDF:
            # The spec allows junk before the header; readers accept it within the first 1 KB.
            if b"%PDF-" not in head[:1024]:
                raise _mismatch(file_type)
        case FileType.DOCX:
            _check_docx(path, settings)
        case FileType.TXT | FileType.MD:
            has_utf16_bom = head.startswith((b"\xff\xfe", b"\xfe\xff"))
            if head.startswith(_BINARY_SIGNATURES) or (b"\x00" in head and not has_utf16_bom):
                raise _mismatch(file_type)
