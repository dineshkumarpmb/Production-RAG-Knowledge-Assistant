import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

SCHEME = "local://"


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return path

    def _key(self, uri: str) -> str:
        if not uri.startswith(SCHEME):
            raise ValueError(f"Not a local storage URI: {uri!r}")
        return uri.removeprefix(SCHEME)

    def put_file(self, key: str, src: Path) -> str:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return SCHEME + key

    @contextmanager
    def local_path(self, uri: str) -> Iterator[Path]:
        path = self._path(self._key(uri))
        if not path.is_file():
            raise FileNotFoundError(f"Stored file is missing: {uri}")
        yield path

    def delete(self, uri: str) -> None:
        self._path(self._key(uri)).unlink(missing_ok=True)


@lru_cache
def get_storage() -> LocalStorage:
    return LocalStorage(get_settings().storage_dir)
