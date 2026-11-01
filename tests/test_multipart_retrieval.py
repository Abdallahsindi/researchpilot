"""Regression tests for multi-part planning and document-source coverage."""

from pathlib import Path

import pytest

from app.agents.planner import plan_research
from app.graph.workflow import merge_document_outcomes
from app.rag.embeddings import HashingEmbeddings
from app.rag.models import DocumentChunk, RetrievedChunk
from app.rag.retrieval import DocumentLibrary
from app.schemas.research import DocumentSearchOutcome, ResearchRequest, SearchQuery

LECTURE_QUESTION = (
    "Explain the main components of a microprocessor, describe the instruction execution "
    "cycle, and explain the difference between signed and unsigned integers. Support each "
    "answer with citations to the relevant PDF pages. If the document does not provide enough "
    "evidence, clearly state that."
)
LECTURE_PDF = (
    Path(__file__).resolve().parents[1] / "data" / "documents" / "files" / "doc_2f11b273f262.pdf"
)


def test_multipart_question_plans_each_topic_separately():
    plan = plan_research(
        ResearchRequest(question=LECTURE_QUESTION),
        max_queries=5,
    )

    assert [query.text for query in plan.queries] == [
        LECTURE_QUESTION,
        "main components of a microprocessor",
        "instruction execution cycle",
        "signed integers",
        "unsigned integers",
    ]
    assert [query.id for query in plan.queries] == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_multipart_selection_takes_one_passage_per_query_before_filling_by_score():
    outcomes = [
        DocumentSearchOutcome(
            query=SearchQuery(id="Q1", text="components", purpose="part 1"),
            chunks=[
                _retrieved("Q1", page=1, text="microprocessor components", score=0.9),
                _retrieved("Q1", page=4, text="extra", score=0.8),
            ],
        ),
        DocumentSearchOutcome(
            query=SearchQuery(id="Q2", text="cycle", purpose="part 2"),
            chunks=[_retrieved("Q2", page=2, text="instruction execution cycle", score=0.7)],
        ),
        DocumentSearchOutcome(
            query=SearchQuery(id="Q3", text="integers", purpose="part 3"),
            chunks=[_retrieved("Q3", page=3, text="signed and unsigned integers", score=0.6)],
        ),
    ]

    selected = merge_document_outcomes(outcomes, max_sources=3)

    assert [item.chunk.page_number for item, _, _ in selected] == [1, 2, 3]
    assert len({query_ids[0] for _, _, query_ids in selected}) == 3


@pytest.mark.skipif(
    not LECTURE_PDF.exists(),
    reason="the uploaded lecture PDF is not present in this checkout",
)
def test_actual_lecture_multipart_queries_retrieve_each_topic(tmp_path: Path):
    # Hashing embeddings keep this regression model-free and deterministic. The separate
    # selection test above covers the production merge; this test uses the real PDF text.
    library = DocumentLibrary(tmp_path / "documents", HashingEmbeddings(), min_score=0.1)
    info, _ = library.ingest(
        LECTURE_PDF.read_bytes(),
        "Lec 1 General Concepts and Data Representation.pdf",
    )
    plan = plan_research(
        ResearchRequest(question=LECTURE_QUESTION, document_ids=[info.document_id]),
        max_queries=5,
    )
    outcomes = [
        DocumentSearchOutcome(
            query=query,
            chunks=library.retrieve(query.text, top_k=3, document_ids=[info.document_id]),
        )
        for query in plan.queries
    ]
    text = " ".join(item.chunk.text.casefold() for outcome in outcomes for item in outcome.chunks)

    assert "arithmetic" in text and "control unit" in text
    assert all(term in text for term in ("fetch", "decode", "execute"))
    assert "signed" in text and "unsigned" in text


def _retrieved(query_id: str, *, page: int, text: str, score: float) -> RetrievedChunk:
    chunk = DocumentChunk(
        chunk_id=f"doc_multipart:p{page}:c0",
        document_id="doc_multipart",
        filename="lecture.pdf",
        page_number=page,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=1)
