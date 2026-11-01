"""Data models shared by the API, the workflow, and storage."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.rag.models import RetrievedChunk
from app.schemas.analysis import AnalysisResult

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ResearchRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1000)
    ]
    goals: list[ShortText] = Field(default_factory=list, max_length=10)
    constraints: list[ShortText] = Field(default_factory=list, max_length=10)
    criteria: list[ShortText] = Field(
        default_factory=list,
        max_length=10,
        description="Comparison criteria, e.g. 'cost', 'latency', 'ease of setup'.",
    )
    document_ids: list[Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{12}$")]] = Field(
        default_factory=list,
        max_length=20,
        description="Uploaded documents (from POST /documents) to search alongside the web.",
    )
    analyze: bool = Field(
        default=False,
        description="Run the Analyst, Verifier, and Report Writer on the collected sources "
        "using the local LLM. Slower; the model loads on first use.",
    )


class SearchQuery(BaseModel):
    id: str
    text: str
    purpose: str


class ResearchPlan(BaseModel):
    objective: str
    queries: list[SearchQuery]
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    """A raw result returned by a search provider."""

    url: str
    title: str
    snippet: str = ""
    score: float | None = None
    published_date: str | None = None


class SearchOutcome(BaseModel):
    """The result of running one planned query, successful or not."""

    query: SearchQuery
    provider: str
    hits: list[SearchHit] = Field(default_factory=list)
    error: str | None = None
    attempts: int = 1
    duration_ms: float = 0.0
    retrieved_at: datetime = Field(default_factory=utc_now)

    @property
    def ok(self) -> bool:
        return self.error is None


class DocumentSearchOutcome(BaseModel):
    """The result of searching uploaded documents for one planned query."""

    query: SearchQuery
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    # Raw top-k hits before thresholding, plus the counts needed to explain an empty result.
    candidates: list[RetrievedChunk] = Field(default_factory=list)
    indexed_chunks: int = 0
    selected_chunks: int = 0
    selected_document_ids: list[str] = Field(default_factory=list)
    min_score: float = 0.0
    fallback_used: bool = False
    rejection_reason: str | None = None
    error: str | None = None
    duration_ms: float = 0.0
    retrieved_at: datetime = Field(default_factory=utc_now)

    @property
    def ok(self) -> bool:
        return self.error is None


DEMO_DOMAIN = "demo.example"


def is_demo_url(url: str | None) -> bool:
    return bool(url) and urlsplit(url).netloc.lower() == DEMO_DOMAIN


class SourceType(StrEnum):
    WEB = "web"  # a real page returned by a live search provider
    DOCUMENT = "document"  # a chunk of a PDF the user uploaded
    DEMO = "demo"  # synthetic Demo-mode corpus entry; never real evidence


class IndexedDocument(BaseModel):
    """A selected document and how much of it is in the active index."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentIndexDiagnostics(BaseModel):
    """The index state and document selection used by one run."""

    indexed_chunks: int
    selected_documents: list[IndexedDocument] = Field(default_factory=list)


class DocumentCandidate(BaseModel):
    """One raw vector-search hit, before the similarity threshold was applied."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    score: float
    accepted: bool


class DocumentSearchDiagnostics(BaseModel):
    """Why a document query produced the passages it did (or none)."""

    query_id: str
    query: str
    selected_document_ids: list[str]
    indexed_chunks: int
    selected_chunks: int
    min_score: float
    fallback_used: bool = False
    accepted: int = 0
    rejected: int = 0
    candidates: list[DocumentCandidate] = Field(default_factory=list)
    reason: str | None = None


class Source(BaseModel):
    """A deduplicated, ranked source that findings can cite by `id` (S1, S2, ...)."""

    id: str
    source_type: SourceType
    url: str | None = None
    domain: str | None = None
    title: str
    snippet: str
    provider: str
    score: float | None = None
    published_date: str | None = None
    retrieved_at: datetime
    query_ids: list[str] = Field(description="Planned queries that surfaced this source.")
    # Document sources only
    document_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_type(cls, data):
        # Runs stored before source types existed only contain web or demo sources.
        if isinstance(data, dict) and "source_type" not in data:
            demo = data.get("provider") == "demo" or is_demo_url(data.get("url"))
            data = {**data, "source_type": SourceType.DEMO if demo else SourceType.WEB}
        return data


class RunStatus(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class SearchError(BaseModel):
    query_id: str
    query: str
    message: str
    attempts: int


class RunMetrics(BaseModel):
    duration_ms: float
    search_calls: int = Field(description="Provider calls, including retries.")
    failed_queries: int
    raw_hits: int
    sources_returned: int
    document_searches: int = 0
    document_chunks_retrieved: int = 0
    document_candidates_retrieved: int = 0
    document_candidates_rejected: int = 0
    llm_calls: int = 0


class ResearchResult(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    status: RunStatus
    mode: str = Field(description="'demo' or 'live'.")
    search_provider: str
    request: ResearchRequest
    plan: ResearchPlan
    sources: list[Source]
    warnings: list[str] = Field(default_factory=list)
    errors: list[SearchError] = Field(default_factory=list)
    metrics: RunMetrics
    document_index: DocumentIndexDiagnostics | None = Field(
        default=None, description="Index state and selected documents, when PDFs were searched."
    )
    document_retrieval: list[DocumentSearchDiagnostics] = Field(
        default_factory=list,
        description="Per-query document candidates, scores, and rejection reasons.",
    )
    analysis: AnalysisResult | None = Field(
        default=None, description="Analyst attempts and verification; present when analyze=true."
    )
    report: str | None = Field(default=None, description="Markdown report when analyze=true.")
    created_at: datetime = Field(default_factory=utc_now)
