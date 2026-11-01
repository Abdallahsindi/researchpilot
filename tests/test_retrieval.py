import json
import sys

import numpy as np
import pytest

from app.rag.chunking import chunk_document
from app.rag.deps import MissingDependencyError
from app.rag.embeddings import HashingEmbeddings, SentenceTransformerEmbeddings
from app.rag.models import DocumentChunk
from app.rag.pdf import NoExtractableTextError, parse_pdf
from app.rag.retrieval import DocumentLibrary, UnknownDocumentError, VectorRetriever
from app.rag.vector_store import (
    FaissVectorStore,
    IndexNotFoundError,
    StaleIndexError,
    VectorStoreError,
)
from tests.pdf_factory import build_pdf


def _chunk(chunk_id: str, document_id: str = "doc_a", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page,
        chunk_index=0,
        text=chunk_id,
        start_char=0,
        end_char=len(chunk_id),
    )


def _chunk_with_text(text: str, document_id: str = "doc_a", page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{document_id}:p{page}:c0",
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_hashing_embeddings_shape_and_normalization():
    embedder = HashingEmbeddings(dimension=64)

    matrix = embedder.embed_documents(["solar power", "battery storage", ""])

    assert matrix.shape == (3, 64) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix[:2], axis=1), 1.0)
    assert not matrix[2].any()  # empty text -> zero vector, no NaNs
    assert np.array_equal(embedder.embed_query("solar power"), matrix[0])  # deterministic
    assert embedder.embed_documents([]).shape == (0, 64)


def test_faiss_ranks_by_cosine_similarity():
    store = FaissVectorStore(dimension=3)
    vectors = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)
    store.add([_chunk("x"), _chunk("xy"), _chunk("z")], vectors)

    results = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["x", "xy"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def test_faiss_document_filter_and_delete():
    store = FaissVectorStore(dimension=2)
    store.add(
        [_chunk("a1", "doc_a"), _chunk("b1", "doc_b"), _chunk("a2", "doc_a")],
        np.array([[1, 0], [1, 0.1], [0, 1]], dtype=np.float32),
    )

    filtered = store.search(np.array([1, 0], dtype=np.float32), top_k=5, document_ids={"doc_b"})
    assert [r.chunk.chunk_id for r in filtered] == ["b1"]

    store.delete_document("doc_a")
    assert len(store) == 1 and store.document_ids() == {"doc_b"}
    assert store.search(np.array([0, 1], dtype=np.float32), top_k=5)[0].chunk.chunk_id == "b1"


def test_faiss_rejects_mismatched_shapes():
    store = FaissVectorStore(dimension=4)
    with pytest.raises(VectorStoreError, match="shape"):
        store.add([_chunk("a")], np.ones((1, 3), dtype=np.float32))
    with pytest.raises(VectorStoreError, match="1 chunks but 2"):
        store.add([_chunk("a")], np.ones((2, 4), dtype=np.float32))
    assert store.search(np.ones(4, dtype=np.float32), top_k=3) == []


def test_faiss_save_load_and_stale_detection(tmp_path):
    store = FaissVectorStore(dimension=2)
    store.add([_chunk("a"), _chunk("b")], np.array([[1, 0], [0, 1]], dtype=np.float32))
    manifest = {"embedding_model": "m1", "chunk_size": 800}
    store.save(tmp_path, manifest)

    loaded = FaissVectorStore.load(tmp_path, manifest)
    assert [c.chunk_id for c in loaded.chunks] == ["a", "b"]
    assert loaded.search(np.array([0, 1], dtype=np.float32), 1)[0].chunk.chunk_id == "b"

    with pytest.raises(StaleIndexError, match="embedding_model"):
        FaissVectorStore.load(tmp_path, {**manifest, "embedding_model": "m2"})
    (tmp_path / "chunks.jsonl").write_text("")
    with pytest.raises(StaleIndexError, match="out of sync"):
        FaissVectorStore.load(tmp_path, manifest)
    with pytest.raises(IndexNotFoundError):
        FaissVectorStore.load(tmp_path / "missing", manifest)


def test_retriever_ranks_relevant_page_first_and_filters_irrelevant(sample_pdf):
    embedder = HashingEmbeddings()
    store = FaissVectorStore(embedder.dimension)
    chunks = chunk_document(parse_pdf(sample_pdf, "report.pdf"))
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = VectorRetriever(embedder, store, min_score=0.1)

    battery = retriever.retrieve("Why was lithium iron phosphate chosen for the battery?", top_k=3)
    cost = retriever.retrieve("What was the total installed cost?", top_k=3)

    assert battery[0].chunk.page_number == 2
    assert cost[0].chunk.page_number == 4
    assert [r.rank for r in battery] == list(range(1, len(battery) + 1))
    assert all(r.score >= 0.1 for r in battery + cost)
    assert retriever.retrieve("sourdough bread starter recipe", top_k=3) == []


def test_retriever_fallback_requires_a_distinctive_lexical_anchor():
    embedder = HashingEmbeddings(dimension=64)
    store = FaissVectorStore(embedder.dimension)
    text = "voltage regulator limits current and protects the circuit"
    store.add([_chunk_with_text(text)], embedder.embed_documents([text]))
    retriever = VectorRetriever(embedder, store, min_score=0.99, fallback_min_score=0.02)

    anchored = retriever.retrieve_with_diagnostics("voltage regulator", top_k=1)
    unrelated = retriever.retrieve_with_diagnostics(
        "what is the airspeed velocity of an unladen swallow", top_k=1
    )
    framing_only = retriever.retrieve_with_diagnostics("main results second", top_k=1)

    # The normal threshold is still the primary rule; the fallback only admits a passage
    # that shares distinctive subject wording with the question.
    assert anchored.fallback_used and len(anchored.chunks) == 1
    assert unrelated.chunks == [] and not unrelated.fallback_used
    assert framing_only.chunks == [] and not framing_only.fallback_used
    assert unrelated.candidates and "similarity threshold" in unrelated.rejection_reason


def test_retrieval_diagnostics_keep_rejected_candidates():
    embedder = HashingEmbeddings(dimension=64)
    store = FaissVectorStore(embedder.dimension)
    text = "battery storage uses lithium iron phosphate cells"
    store.add([_chunk_with_text(text)], embedder.embed_documents([text]))
    retriever = VectorRetriever(embedder, store, min_score=0.99)

    result = retriever.retrieve_with_diagnostics("lithium battery", top_k=1)

    assert result.chunks == []
    assert len(result.candidates) == 1 and result.indexed_chunks == 1
    assert "similarity threshold 0.99" in result.rejection_reason
    assert f"{result.candidates[0].score:.2f}" in result.rejection_reason


def test_library_ingests_dedupes_and_persists(tmp_path, sample_pdf):
    library = DocumentLibrary(tmp_path, HashingEmbeddings(), min_score=0.1)

    info, created = library.ingest(sample_pdf, "report.pdf")
    again, created_again = library.ingest(sample_pdf, "copy.pdf")

    assert created and not created_again and again == info
    assert info.page_count == 4 and info.pages_with_text == 3 and info.chunk_count == 3
    assert (tmp_path / "files" / f"{info.document_id}.pdf").read_bytes() == sample_pdf
    assert json.loads((tmp_path / "faiss" / "manifest.json").read_text())["document_ids"] == [
        info.document_id
    ]

    reopened = DocumentLibrary(tmp_path, HashingEmbeddings(), min_score=0.1)
    assert [d.document_id for d in reopened.list_documents()] == [info.document_id]
    assert reopened.retrieve("battery chemistry lithium", top_k=1)[0].chunk.page_number == 2


def test_library_rebuilds_stale_or_missing_index_from_pdfs(tmp_path, sample_pdf):
    info, _ = DocumentLibrary(tmp_path, HashingEmbeddings()).ingest(sample_pdf, "report.pdf")

    # Different embedding model -> manifest mismatch -> rebuild with the new model.
    rebuilt = DocumentLibrary(tmp_path, HashingEmbeddings(dimension=256), min_score=0.1)
    assert len(rebuilt.store) == info.chunk_count
    assert rebuilt.retrieve("battery lithium", top_k=1)[0].chunk.page_number == 2

    # Deleted index files -> rebuild from the stored PDFs.
    for path in (tmp_path / "faiss").iterdir():
        path.unlink()
    restored = DocumentLibrary(tmp_path, HashingEmbeddings(dimension=256))
    assert len(restored.store) == info.chunk_count


def test_library_rejects_unknown_documents_and_unreadable_pdfs(tmp_path):
    library = DocumentLibrary(tmp_path, HashingEmbeddings())

    with pytest.raises(UnknownDocumentError):
        library.retrieve("anything", document_ids=["doc_000000000000"])
    with pytest.raises(NoExtractableTextError):
        library.ingest(build_pdf([None]), "scan.pdf")
    assert library.list_documents() == []
    assert not list((tmp_path / "files").iterdir())


def test_missing_faiss_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)

    with pytest.raises(MissingDependencyError, match="pip install faiss-cpu"):
        FaissVectorStore(dimension=8)


def test_missing_sentence_transformers_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbeddings()  # constructing never loads the model

    with pytest.raises(MissingDependencyError, match="sentence-transformers"):
        embedder.embed_query("hello")
