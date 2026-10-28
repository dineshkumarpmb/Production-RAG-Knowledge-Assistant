from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    """Where original uploads live (a local Docker volume; the project runs locally only)."""

    def put_file(self, key: str, src: Path) -> str:
        """Store `src` under `key` and return its URI."""
        ...

    def local_path(self, uri: str) -> AbstractContextManager[Path]:
        """Yield a local filesystem path for the object (downloading it first if remote)."""
        ...

    def delete(self, uri: str) -> None: ...
