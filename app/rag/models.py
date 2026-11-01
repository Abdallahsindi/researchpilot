"""Data models for documents, chunks, and retrieval results."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int = Field(description="1-based page number in the original PDF.")
    text: str

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


class ParsedDocument(BaseModel):
    document_id: str = Field(description="Stable id derived from the file's SHA-256 hash.")
    filename: str
    sha256: str
    size_bytes: int
    page_count: int
    pages: list[PageText]
    metadata: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def pages_with_text(self) -> int:
        return sum(1 for page in self.pages if page.has_text)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="'<document_id>:p<page>:c<index>', stable for a given PDF.")
    document_id: str
    filename: str
    page_number: int
    chunk_index: int = Field(description="Position of the chunk within its page.")
    text: str
    start_char: int = Field(description="Offset of the chunk within the page text.")
    end_char: int
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float = Field(description="Cosine similarity between query and chunk, -1..1.")
    rank: int


class RetrievalResult(BaseModel):
    """Vector-search output, including candidates rejected by the similarity threshold.

    `chunks` is what callers may use as evidence. `candidates` is the raw top-k result so
    callers and the API can explain an empty result instead of silently returning nothing.
    """

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    candidates: list[RetrievedChunk] = Field(default_factory=list)
    indexed_chunks: int = 0
    selected_chunks: int = 0
    selected_document_ids: list[str] = Field(default_factory=list)
    min_score: float = 0.0
    fallback_used: bool = False
    rejection_reason: str | None = None


class DocumentInfo(BaseModel):
    """Summary of an ingested document, returned by the API."""

    document_id: str
    filename: str
    sha256: str
    size_bytes: int
    page_count: int
    pages_with_text: int
    chunk_count: int
    metadata: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
