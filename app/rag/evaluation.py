"""Small evaluation harness for retrieval and grounded QA.

Each case states whether the document can answer the question and, if so, which page(s)
contain the evidence. Metrics are computed from actual runs; nothing is estimated.
"""

from collections.abc import Callable
from statistics import mean

from pydantic import BaseModel, Field

from app.rag.qa import AnswerStatus, QAResult, Retriever


class EvalCase(BaseModel):
    id: str
    question: str
    answerable: bool
    expected_pages: list[int] = Field(default_factory=list)


class RetrievalCaseResult(BaseModel):
    case_id: str
    retrieved_pages: list[int]
    top_score: float | None
    hit: bool | None = Field(description="Expected page in top-k; None for unanswerable cases.")


class QACaseResult(BaseModel):
    case_id: str
    answerable: bool
    status: AnswerStatus
    correct_decision: bool = Field(description="Answered when answerable, abstained otherwise.")
    cited_pages: list[int]
    cited_expected_page: bool | None
    valid_citations: int
    invalid_citations: int
    evidence_support: float | None
    latency_ms: float


def evaluate_retrieval(
    retriever: Retriever, cases: list[EvalCase], top_k: int = 4
) -> list[RetrievalCaseResult]:
    results = []
    for case in cases:
        retrieved = retriever.retrieve(case.question, top_k=top_k)
        pages = [r.chunk.page_number for r in retrieved]
        hit = bool(set(pages) & set(case.expected_pages)) if case.answerable else None
        top = retrieved[0].score if retrieved else None
        results.append(
            RetrievalCaseResult(case_id=case.id, retrieved_pages=pages, top_score=top, hit=hit)
        )
    return results


def evaluate_qa(answer_fn: Callable[[str], QAResult], cases: list[EvalCase]) -> list[QACaseResult]:
    results = []
    for case in cases:
        result = answer_fn(case.question)
        answered = result.status is AnswerStatus.ANSWERED
        pages = sorted({c.page_number for c in result.supporting_citations})
        cited_expected = (
            bool(set(pages) & set(case.expected_pages)) if case.answerable and answered else None
        )
        results.append(
            QACaseResult(
                case_id=case.id,
                answerable=case.answerable,
                status=result.status,
                correct_decision=answered == case.answerable,
                cited_pages=pages,
                cited_expected_page=cited_expected,
                valid_citations=len(result.citations),
                invalid_citations=len(result.invalid_citations),
                evidence_support=result.evidence_support,
                latency_ms=result.metrics.retrieval_ms + result.metrics.generation_ms,
            )
        )
    return results


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(
    retrieval: list[RetrievalCaseResult], qa: list[QACaseResult] | None = None
) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {
        "cases": len(retrieval),
        "retrieval_hit_rate": _rate([r.hit for r in retrieval if r.hit is not None]),
    }
    if qa:
        valid = sum(r.valid_citations for r in qa)
        total = valid + sum(r.invalid_citations for r in qa)
        support = [r.evidence_support for r in qa if r.evidence_support is not None]
        summary.update(
            decision_accuracy=_rate([r.correct_decision for r in qa]),
            cited_expected_page_rate=_rate(
                [r.cited_expected_page for r in qa if r.cited_expected_page is not None]
            ),
            citation_validity=round(valid / total, 3) if total else None,
            mean_evidence_support=round(mean(support), 3) if support else None,
            mean_latency_ms=round(mean(r.latency_ms for r in qa), 1),
        )
    return summary
