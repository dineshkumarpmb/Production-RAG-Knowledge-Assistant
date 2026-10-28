from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables and `.env`."""

    # env_ignore_empty: Compose passes unset variables as "", which should mean "use the default".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    env: str = "dev"  # dev | test | prod
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"
    redis_url: str = "redis://localhost:6379/0"

    storage_dir: Path = Path("./data/uploads")

    # Upload limits
    max_upload_bytes: int = 50 * 1024 * 1024
    max_docx_uncompressed_bytes: int = 200 * 1024 * 1024

    # Parsing
    low_text_page_chars: int = 50  # pages with less text are flagged as OCR candidates
    docx_virtual_page_chars: int = 3000  # target size of a DOCX "virtual page"

    # Chunking (tuned in Phase 9)
    chunk_target_tokens: int = 500
    chunk_overlap_ratio: float = 0.15
    chunk_contextual_header: bool = True  # embed "title > heading path" with each chunk

    # Embeddings. Both default models produce 1024-dim vectors (see db.models.EMBEDDING_DIM).
    embedding_provider: Literal["voyage", "local"] = "voyage"
    embedding_model: str | None = None  # None: voyage-4 / BAAI/bge-m3 depending on the provider
    embedding_price_per_million_tokens: float | None = None  # None: provider default
    voyage_api_key: SecretStr | None = None
    voyage_base_url: str = "https://api.voyageai.com/v1"
    embedding_batch_size: int = 128
    embedding_max_batch_tokens: int = 100_000  # Voyage allows 320K for voyage-4; stay well under
    embedding_max_retries: int = 5
    embedding_requests_per_minute: int = 300
    embedding_timeout_seconds: float = 60.0

    # Near-duplicate documents: SimHash Hamming distance (out of 64 bits) at or below this
    near_duplicate_max_distance: int = 3

    # Retrieval (Phase 3)
    retrieval_candidates: int = 50  # top-N from each of keyword and vector search
    retrieval_rrf_k: int = 60
    retrieval_keyword_weight: float = 1.0
    retrieval_vector_weight: float = 1.0
    hnsw_ef_search: int = 100  # higher = better recall, slower; must be >= candidates

    # Reranking (Phase 4)
    reranker_provider: Literal["local", "voyage", "none"] = "local"
    reranker_model: str | None = None  # None: provider default (see app.rerank)
    reranker_price_per_million_tokens: float | None = None  # None: provider default
    rerank_candidates: int = 30  # fused results the reranker scores
    rerank_top_k: int = 8  # chunks kept for the answer context
    rerank_min_score: float | None = None  # best score below this: "not found in your documents"
    rerank_max_length: int = 512  # tokens per (query, chunk) pair for local cross-encoders
    context_token_budget: int = 3000

    # Answer generation with Claude (Phase 5). The SDK also reads ANTHROPIC_API_KEY itself.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000  # thinking counts toward this as well as the answer
    llm_fallbacks: bool = True  # re-run refused requests on Anthropic's recommended model
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    # Per-model prices in USD per million tokens, overriding app.generation.pricing, e.g.
    # {"claude-opus-5-5": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5}}
    llm_prices: dict[str, dict[str, float]] = {}

    # Local models. With MODEL_SERVER_URL set, the API and worker call one shared model
    # server (the `models` service) instead of each loading its own copy of the models.
    model_server_url: str | None = None
    model_server_timeout_seconds: float = 120.0
    local_model_device: str | None = None  # "cpu" / "cuda"; None picks automatically
    local_model_dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    # Placeholder identity used until authentication lands in Phase 6.
    dev_tenant_name: str = "default"
    dev_user_email: str = "dev@example.com"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()
