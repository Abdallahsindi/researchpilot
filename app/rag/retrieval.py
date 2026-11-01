"""Retrieval over embedded chunks, plus a persistent document library for the API."""

import logging
import os
import re
import tempfile
import threading
from collections.abc import Collection
from pathlib import Path

from app.rag.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_document
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import (
    DocumentChunk,
    DocumentInfo,
    ParsedDocument,
    RetrievalResult,
    RetrievedChunk,
)
from app.rag.pdf import DEFAULT_MAX_BYTES, document_id_for, parse_pdf
from app.rag.vector_store import FaissVectorStore, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
# Framing words do not identify a subject. They must not authorize a below-threshold
# fallback on their own; e.g. "results" or "second" can appear in an unrelated document.
_FALLBACK_IGNORED = frozenset(
    {
        "about",
        "according",
        "after",
        "answer",
        "based",
        "briefly",
        "citation",
        "citations",
        "cite",
        "could",
        "describe",
        "detail",
        "difference",
        "document",
        "documents",
        "each",
        "evidence",
        "explain",
        "from",
        "give",
        "have",
        "identify",
        "into",
        "list",
        "main",
        "many",
        "more",
        "most",
        "page",
        "pages",
        "pdf",
        "please",
        "provide",
        "question",
        "relevant",
        "results",
        "second",
        "should",
        "source",
        "sources",
        "state",
        "steps",
        "summarize",
        "support",
        "tell",
        "than",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "using",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }
)


def _word_forms(word: str) -> set[str]:
    """Cheap inflection folding for the fallback anchor (fetching/fetch, cycles/cycle)."""
    forms = {word}
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            forms.add(word[: -len(suffix)])
    return forms


def _fallback_anchor(query: str, text: str) -> bool:
    """Whether a candidate shares enough distinctive wording with the question.

    A semantic score can dip below the normal floor for a short or paraphrased question.
    The fallback only admits a candidate when at least two subject terms overlap (or one
    long, distinctive term does). Framing words alone never qualify.
    """
    query_terms = {
        word
        for word in _WORD_RE.findall(query.lower())
        if len(word) >= 5 and word not in _FALLBACK_IGNORED and not word.isdigit()
    }
    if not query_terms:
        return False
    text_forms = set()
    for word in _WORD_RE.findall(text.lower()):
        text_forms.update(_word_forms(word))
    matches = {term for term in query_terms if _word_forms(term) & text_forms}
    return len(matches) >= 2 or any(len(term) >= 8 for term in matches)


class VectorRetriever:
    """Embeds a query and returns the most similar chunks above `min_score`."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        *,
        min_score: float = 0.0,
        fallback_min_score: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.min_score = min_score
        self.fallback_min_score = fallback_min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Collection[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_with_diagnostics(query, top_k, document_ids, min_score).chunks

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Collection[str] | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        threshold = self.min_score if min_score is None else min_score
        candidates = self.store.search(self.embeddings.embed_query(query), top_k, document_ids)
        kept = [r for r in candidates if r.score >= threshold]
        fallback_used = False
        reason: str | None = None

        fallback = self.fallback_min_score
        if not kept and fallback is not None and fallback < threshold:
            # This is deliberately conservative: a low semantic score is accepted only when
            # the passage shares distinctive wording with the question. It is a safety net,
            # not a replacement for the normal threshold.
            eligible = [
                r
                for r in candidates
                if fallback <= r.score < threshold and _fallback_anchor(query, r.chunk.text)
            ]
            if eligible:
                kept = [eligible[0]]
                fallback_used = True

        if not kept:
            if not candidates:
                reason = "No indexed candidates were returned for the selected documents."
            else:
                best = candidates[0]
                reason = (
                    f"No candidate met the similarity threshold {threshold:.2f}; "
                    f"the best score was {best.score:.2f} "
                    f"({best.chunk.filename}, page {best.chunk.page_number})."
                )
                if fallback is not None and fallback < threshold and best.score >= fallback:
                    reason += (
                        " The calibrated fallback also rejected it because it shared too few "
                        "distinctive terms with the question."
                    )

        chunks = [r.model_copy(update={"rank": rank}) for rank, r in enumerate(kept, start=1)]
        return RetrievalResult(
            chunks=chunks,
            candidates=candidates,
            indexed_chunks=len(self.store),
            min_score=threshold,
            fallback_used=fallback_used,
            rejection_reason=reason,
        )


def index_document(
    document: ParsedDocument,
    embeddings: EmbeddingProvider,
    store: VectorStore,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    chunks = chunk_document(document, chunk_size, chunk_overlap)
    store.add(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return chunks


class UnknownDocumentError(KeyError):
    def __init__(self, document_ids: Collection[str]) -> None:
        super().__init__(", ".join(sorted(document_ids)))
        self.document_ids = sorted(document_ids)


class DocumentLibrary:
    """Stores uploaded PDFs and keeps a vector index of their chunks.

    Layout under `root`:
        files/<document_id>.pdf    original upload (source of truth)
        files/<document_id>.json   DocumentInfo
        faiss/                     index, chunk metadata, manifest (FAISS backend only)

    The FAISS index is derived data. If it is missing, corrupt, or was built with a different
    embedding model or chunking settings, it is rebuilt from the stored PDFs.
    """

    def __init__(
        self,
        root: Path,
        embeddings: EmbeddingProvider,
        store: VectorStore | None = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_file_bytes: int = DEFAULT_MAX_BYTES,
        min_score: float = 0.0,
        fallback_min_score: float | None = None,
    ) -> None:
        self.root = root
        self.files_dir = root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_bytes = max_file_bytes
        self._lock = threading.Lock()
        self._documents = self._load_document_infos()
        # An external store (pgvector) persists itself; otherwise use a local FAISS index.
        self._faiss_dir = None if store is not None else root / "faiss"
        self.store = store if store is not None else self._open_faiss_index()
        self.retriever = VectorRetriever(
            embeddings,
            self.store,
            min_score=min_score,
            fallback_min_score=fallback_min_score,
        )

    def _load_document_infos(self) -> dict[str, DocumentInfo]:
        infos = {}
        for path in sorted(self.files_dir.glob("*.json")):
            info = DocumentInfo.model_validate_json(path.read_text(encoding="utf-8"))
            if (self.files_dir / f"{info.document_id}.pdf").exists():
                infos[info.document_id] = info
        return infos

    def _manifest(self) -> dict:
        return {
            "embedding_model": self.embeddings.model_id,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "document_ids": sorted(self._documents),
        }

    def _open_faiss_index(self) -> FaissVectorStore:
        assert self._faiss_dir is not None
        try:
            store = FaissVectorStore.load(self._faiss_dir, self._manifest())
            logger.info("Loaded FAISS index with %d chunks", len(store))
            return store
        except VectorStoreError as exc:
            if not self._documents:
                return FaissVectorStore(self.embeddings.dimension)
            logger.warning("Rebuilding FAISS index from %d PDFs: %s", len(self._documents), exc)
            return self._rebuild()

    def _rebuild(self) -> FaissVectorStore:
        store = FaissVectorStore(self.embeddings.dimension)
        for document_id, info in self._documents.items():
            data = (self.files_dir / f"{document_id}.pdf").read_bytes()
            document = parse_pdf(data, info.filename, max_bytes=max(len(data), 1))
            index_document(
                document,
                self.embeddings,
                store,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        self._save_index(store)
        return store

    def _save_index(self, store: VectorStore) -> None:
        if self._faiss_dir is not None and isinstance(store, FaissVectorStore):
            store.save(self._faiss_dir, self._manifest())

    def _write_atomic(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)

    def ingest(self, data: bytes, filename: str) -> tuple[DocumentInfo, bool]:
        """Parse, chunk, embed, and index a PDF. Returns (info, created)."""
        document_id, _ = document_id_for(data)
        with self._lock:
            if document_id in self._documents:
                return self._documents[document_id], False

            document = parse_pdf(data, filename, max_bytes=self.max_file_bytes)
            # Clear any leftovers (e.g. pgvector rows from an interrupted upload).
            self.store.delete_document(document_id)
            chunks = index_document(
                document,
                self.embeddings,
                self.store,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            info = DocumentInfo(
                document_id=document.document_id,
                filename=document.filename,
                sha256=document.sha256,
                size_bytes=document.size_bytes,
                page_count=document.page_count,
                pages_with_text=document.pages_with_text,
                chunk_count=len(chunks),
                metadata=document.metadata,
                warnings=document.warnings,
            )
            self._write_atomic(self.files_dir / f"{document_id}.pdf", data)
            self._write_atomic(
                self.files_dir / f"{document_id}.json", info.model_dump_json(indent=2).encode()
            )
            self._documents[document_id] = info
            self._save_index(self.store)
            logger.info("Ingested %s as %s (%d chunks)", filename, document_id, len(chunks))
            return info, True

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(self._documents.values(), key=lambda info: info.ingested_at)

    def get_document(self, document_id: str) -> DocumentInfo | None:
        return self._documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """The stored chunk, used by the Verifier to check citation metadata."""
        with self._lock:
            return self.store.get_chunk(chunk_id)

    def check_documents(self, document_ids: Collection[str]) -> None:
        missing = set(document_ids) - set(self._documents)
        if missing:
            raise UnknownDocumentError(missing)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Collection[str] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_with_diagnostics(query, top_k, document_ids, min_score).chunks

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Collection[str] | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Retrieve evidence plus the raw candidates and counts needed to explain the result."""
        if document_ids is not None:
            self.check_documents(document_ids)
        selected_ids = (
            sorted(self._documents) if document_ids is None else sorted(set(document_ids))
        )
        with self._lock:
            result = self.retriever.retrieve_with_diagnostics(query, top_k, document_ids, min_score)
            selected_chunks = sum(
                self._documents[document_id].chunk_count
                for document_id in selected_ids
                if document_id in self._documents
            )
            return result.model_copy(
                update={
                    "selected_document_ids": selected_ids,
                    "selected_chunks": selected_chunks,
                }
            )
