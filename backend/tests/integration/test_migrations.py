from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import sync_url
from app.db.unmapped import include_object
from tests.conftest import requires_postgres

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _is_sqlite_vector_artifact(diff: Any) -> bool:
    """SQLite reflects pgvector's VECTOR(n) as NUMERIC, which is not a real difference."""
    changes = diff if isinstance(diff, list) else [diff]
    return all(c[0] == "modify_type" and isinstance(c[-1], Vector) for c in changes)


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = f"sqlite:///{(tmp_path / 'migrate.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def test_migrations_match_models(db_url: str) -> None:
    """Upgrade, compare the schema with the ORM models, then downgrade cleanly (SQLite)."""
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "app" / "db" / "migrations"))
    config.attributes["configure_logging"] = False

    command.upgrade(config, "head")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert [d for d in diff if not _is_sqlite_vector_artifact(d)] == []

    command.downgrade(config, "base")


@requires_postgres
def test_postgres_schema_matches_models(postgres_url: str | None) -> None:
    """The session fixture built this schema with the real migrations (0001..head)."""
    assert postgres_url
    engine = create_engine(sync_url(postgres_url))
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"include_object": include_object})
        diff = compare_metadata(context, Base.metadata)
        tsv = conn.execute(
            text(
                "SELECT count(*) FROM pg_indexes WHERE indexname IN"
                " ('ix_chunks_tsv', 'ix_chunks_embedding_hnsw')"
            )
        ).scalar()
    engine.dispose()
    assert diff == []
    assert tsv == 2
