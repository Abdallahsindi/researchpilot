"""Builds the document library from application settings."""

from app.config import Settings
from app.rag.embeddings import EmbeddingProvider, HashingEmbeddings, SentenceTransformerEmbeddings
from app.rag.retrieval import DocumentLibrary
from app.rag.vector_store import VectorStore


class DocumentsUnavailableError(RuntimeError):
    """Document features cannot start, e.g. an optional dependency or model is missing."""


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddings()
    return SentenceTransformerEmbeddings(settings.embedding_model)


def build_document_library(settings: Settings) -> DocumentLibrary:
    embeddings = build_embeddings(settings)
    store: VectorStore | None = None
    if settings.vector_store == "pgvector":
        from app.rag.pgvector_store import PgVectorStore

        assert settings.supabase_db_url is not None  # enforced by Settings validation
        pg_store = PgVectorStore(
            settings.supabase_db_url.get_secret_value(),
            embeddings.dimension,
            table=settings.pgvector_table,
        )
        pg_store.ensure_schema()
        store = pg_store
    return DocumentLibrary(
        settings.data_dir / "documents",
        embeddings,
        store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_file_bytes=int(settings.max_upload_mb * 1024 * 1024),
        min_score=settings.document_min_score,
        fallback_min_score=settings.document_fallback_min_score,
    )
