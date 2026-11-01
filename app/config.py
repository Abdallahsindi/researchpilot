"""Application configuration loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # Demo mode runs the full workflow against an offline corpus, so no paid API keys are needed.
    demo_mode: bool = True
    search_provider: Literal["tavily"] = "tavily"
    tavily_api_key: SecretStr | None = None

    max_search_queries: int = Field(default=5, ge=1, le=10)
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    max_sources: int = Field(default=10, ge=1, le=50)
    min_sources: int = Field(default=2, ge=1)
    search_timeout_seconds: float = Field(default=20.0, gt=0)
    search_max_retries: int = Field(default=2, ge=0, le=5)
    search_retry_backoff_seconds: float = Field(default=0.5, ge=0)

    # Document ingestion and retrieval
    max_upload_mb: float = Field(default=20, gt=0)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0)
    # "hashing" is a lexical, model-free baseline for offline development; not semantic.
    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: Literal["faiss", "pgvector"] = "faiss"
    supabase_db_url: SecretStr | None = None
    pgvector_table: str = "document_chunks"
    document_top_k: int = Field(default=3, ge=1, le=20)
    max_document_sources: int = Field(default=6, ge=1, le=50)
    document_min_score: float = Field(default=0.25, ge=-1, le=1)
    # Conservative fallback: when no candidate reaches the normal floor, a passage may still
    # be used if it reaches this lower score and shares distinctive wording with the question.
    # It is disabled when it is not below `document_min_score`.
    document_fallback_min_score: float | None = Field(default=0.18, ge=-1, le=1)

    # Analysis (Analyst / Verifier / Report Writer). The model loads on the first
    # analyze=true request, never at startup.
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_device: str = "auto"
    analysis_max_sources: int = Field(default=6, ge=1, le=20)
    analysis_source_chars: int = Field(default=600, ge=100, le=4000)
    analysis_max_new_tokens: int = Field(default=512, ge=64, le=2048)
    analysis_timeout_seconds: float = Field(default=180, gt=0)
    llm_load_timeout_seconds: float = Field(default=900, gt=0)
    analysis_max_revisions: int = Field(default=1, ge=0, le=2)
    verifier_llm_review: bool = True
    verifier_max_reviews: int = Field(default=15, ge=0, le=50)

    @model_validator(mode="after")
    def _check_live_credentials(self) -> "Settings":
        if not self.demo_mode and self.search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when DEMO_MODE=false")
        if self.min_sources > self.max_sources:
            raise ValueError("MIN_SOURCES cannot exceed MAX_SOURCES")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.vector_store == "pgvector" and not self.supabase_db_url:
            raise ValueError("SUPABASE_DB_URL is required when VECTOR_STORE=pgvector")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
