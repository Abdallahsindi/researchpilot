"""Vector stores. `FaissVectorStore` is the local default; see `pgvector_store` for Supabase."""

import json
import logging
import os
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.deps import require
from app.rag.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 1


class VectorStoreError(RuntimeError):
    pass


class IndexNotFoundError(VectorStoreError):
    pass


class StaleIndexError(VectorStoreError):
    """The persisted index was built with different settings or documents."""


class VectorStore(Protocol):
    def add(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None: ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]: ...

    def delete_document(self, document_id: str) -> None: ...

    def document_ids(self) -> set[str]: ...

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None: ...

    def __len__(self) -> int: ...


def _as_matrix(embeddings: np.ndarray, dimension: int) -> np.ndarray:
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise VectorStoreError(
            f"Expected embeddings of shape (n, {dimension}), got {tuple(embeddings.shape)}"
        )
    return matrix


class FaissVectorStore:
    """Exact cosine-similarity search with a FAISS flat inner-product index.

    A flat index compares the query against every chunk. That is exact and fast enough for
    thousands of chunks, which covers a notebook or a small document library.
    """

    def __init__(self, dimension: int) -> None:
        self._faiss = require("faiss", "local vector search")
        self.dimension = dimension
        self.index = self._faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None:
        matrix = _as_matrix(embeddings, self.dimension).copy()
        if len(chunks) != matrix.shape[0]:
            raise VectorStoreError(f"{len(chunks)} chunks but {matrix.shape[0]} embeddings")
        if not chunks:
            return
        self._faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query = _as_matrix(query_embedding, self.dimension).copy()
        self._faiss.normalize_L2(query)
        # With a document filter, scan everything and filter; exact for a flat index.
        k = len(self.chunks) if document_ids is not None else min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if document_ids is not None and chunk.document_id not in document_ids:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score), rank=len(results) + 1))
            if len(results) == top_k:
                break
        return results

    def delete_document(self, document_id: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.document_id != document_id]
        if len(keep) == len(self.chunks):
            return
        vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
        chunks = [self.chunks[i] for i in keep]
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.add(chunks, vectors)

    def document_ids(self) -> set[str]:
        return {chunk.document_id for chunk in self.chunks}

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def save(self, directory: Path, manifest: dict) -> None:
        """Write the index, chunk metadata, and a manifest describing how it was built."""
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")
        full_manifest = {
            **manifest,
            "format_version": INDEX_FORMAT_VERSION,
            "dimension": self.dimension,
            "chunk_count": len(self.chunks),
        }
        # The manifest is written last, so a crash mid-save leaves no valid manifest behind.
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(full_manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp, directory / "manifest.json")

    @classmethod
    def load(cls, directory: Path, expected: dict) -> "FaissVectorStore":
        """Load a saved index, refusing it if it does not match `expected` manifest fields."""
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise IndexNotFoundError(f"No index manifest in {directory}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise StaleIndexError(f"Unreadable index manifest: {exc}") from exc

        expected = {**expected, "format_version": INDEX_FORMAT_VERSION}
        mismatched = sorted(k for k, v in expected.items() if manifest.get(k) != v)
        if mismatched:
            raise StaleIndexError(f"Index settings changed: {', '.join(mismatched)}")

        store = cls(int(manifest["dimension"]))
        try:
            store.index = store._faiss.read_index(str(directory / "index.faiss"))
            lines = (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            store.chunks = [DocumentChunk.model_validate_json(line) for line in lines if line]
        except Exception as exc:
            raise StaleIndexError(f"Index files are missing or corrupt: {exc}") from exc
        if store.index.ntotal != len(store.chunks) or len(store.chunks) != manifest["chunk_count"]:
            raise StaleIndexError("Index and chunk metadata are out of sync")
        return store
