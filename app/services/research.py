"""Runs the research workflow and turns its final state into a stored `ResearchResult`."""

import logging
import threading
from collections.abc import Callable
from time import perf_counter
from uuid import UUID, uuid4

from app.config import Settings
from app.graph.workflow import build_workflow
from app.logging_config import run_id_var
from app.rag.deps import MissingDependencyError
from app.rag.embeddings import EmbeddingError
from app.rag.models import DocumentChunk, RetrievalResult
from app.rag.retrieval import DocumentLibrary
from app.rag.vector_store import VectorStoreError
from app.schemas.research import (
    DocumentCandidate,
    DocumentIndexDiagnostics,
    DocumentSearchDiagnostics,
    DocumentSearchOutcome,
    IndexedDocument,
    ResearchRequest,
    ResearchResult,
    RunMetrics,
    SearchError,
)
from app.services.documents import DocumentsUnavailableError
from app.services.llm import LLM, HuggingFaceLLM, SerializedLLM
from app.services.storage import JsonRunStore
from app.tools.web_search import SearchProvider

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        settings: Settings,
        provider: SearchProvider,
        store: JsonRunStore,
        library_factory: Callable[[], DocumentLibrary] | None = None,
        llm_factory: Callable[[], LLM] | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.store = store
        self._library_factory = library_factory
        self._library: DocumentLibrary | None = None
        self._library_lock = threading.Lock()
        self._llm_factory = llm_factory or (
            lambda: HuggingFaceLLM(settings.llm_model_id, device=settings.llm_device)
        )
        self._llm: SerializedLLM | None = None
        self._llm_lock = threading.Lock()
        self.workflow = build_workflow(
            provider,
            settings,
            self._retrieve_documents,
            llm_provider=self.llm,
            chunk_lookup=self._lookup_chunk,
        )

    def llm(self) -> SerializedLLM:
        """The shared analysis model, created on the first analyze=true request.

        Creation errors are not cached, so a later request can retry (e.g. after enabling
        internet access). Loading itself happens in the workflow, under a timeout.
        """
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    self._llm = SerializedLLM(self._llm_factory())
        return self._llm

    def _lookup_chunk(self, chunk_id: str) -> DocumentChunk | None:
        try:
            return self.documents().get_chunk(chunk_id)
        except DocumentsUnavailableError:
            return None

    def documents(self) -> DocumentLibrary:
        """The document library, created on first use so the API starts without ML models."""
        if self._library is None:
            with self._library_lock:
                if self._library is None:
                    if self._library_factory is None:
                        raise DocumentsUnavailableError("Document features are not configured.")
                    try:
                        self._library = self._library_factory()
                    except (MissingDependencyError, EmbeddingError, VectorStoreError) as exc:
                        raise DocumentsUnavailableError(str(exc)) from exc
        return self._library

    def _retrieve_documents(
        self, query: str, top_k: int, document_ids: list[str]
    ) -> RetrievalResult:
        return self.documents().retrieve_with_diagnostics(query, top_k, document_ids)

    def _document_index(self, document_ids: list[str]) -> DocumentIndexDiagnostics | None:
        if not document_ids:
            return None
        library = self.documents()
        selected = set(document_ids)
        documents = [
            IndexedDocument(
                document_id=info.document_id,
                filename=info.filename,
                page_count=info.page_count,
                chunk_count=info.chunk_count,
            )
            for info in library.list_documents()
            if info.document_id in selected
        ]
        return DocumentIndexDiagnostics(
            indexed_chunks=len(library.store), selected_documents=documents
        )

    def _document_diagnostics(
        self, outcomes: list[DocumentSearchOutcome]
    ) -> list[DocumentSearchDiagnostics]:
        diagnostics = []
        for outcome in sorted(outcomes, key=lambda item: int(item.query.id[1:])):
            accepted_ids = {item.chunk.chunk_id for item in outcome.chunks}
            candidates = [
                DocumentCandidate(
                    chunk_id=item.chunk.chunk_id,
                    document_id=item.chunk.document_id,
                    filename=item.chunk.filename,
                    page_number=item.chunk.page_number,
                    score=round(item.score, 4),
                    accepted=item.chunk.chunk_id in accepted_ids,
                )
                for item in outcome.candidates
            ]
            diagnostics.append(
                DocumentSearchDiagnostics(
                    query_id=outcome.query.id,
                    query=outcome.query.text,
                    selected_document_ids=outcome.selected_document_ids,
                    indexed_chunks=outcome.indexed_chunks,
                    selected_chunks=outcome.selected_chunks,
                    min_score=outcome.min_score,
                    fallback_used=outcome.fallback_used,
                    accepted=len(accepted_ids),
                    rejected=sum(1 for candidate in candidates if not candidate.accepted),
                    candidates=candidates,
                    reason=outcome.rejection_reason if not outcome.chunks else None,
                )
            )
        return diagnostics

    async def run(self, request: ResearchRequest) -> ResearchResult:
        if request.document_ids:
            # Fail fast (before any searching) if a requested document does not exist.
            self.documents().check_documents(request.document_ids)
        run_id = uuid4()
        token = run_id_var.set(str(run_id))
        started = perf_counter()
        try:
            logger.info("Research run started: %r", request.question)
            state = await self.workflow.ainvoke({"request": request})
            outcomes = state["search_outcomes"]
            document_outcomes = state.get("document_outcomes", [])
            analysis = state.get("analysis")
            llm_calls = (
                analysis.metrics.analyst_calls + analysis.metrics.review_calls if analysis else 0
            )
            document_diagnostics = self._document_diagnostics(document_outcomes)
            result = ResearchResult(
                run_id=run_id,
                status=state["status"],
                mode="demo" if self.settings.demo_mode else "live",
                search_provider=self.provider.name,
                request=request,
                plan=state["plan"],
                sources=state["sources"],
                warnings=state["warnings"],
                errors=[
                    SearchError(
                        query_id=o.query.id,
                        query=o.query.text,
                        message=o.error,
                        attempts=o.attempts,
                    )
                    for o in outcomes
                    if o.error is not None
                ]
                + [
                    SearchError(
                        query_id=o.query.id, query=o.query.text, message=o.error, attempts=1
                    )
                    for o in document_outcomes
                    if o.error is not None
                ],
                metrics=RunMetrics(
                    duration_ms=round((perf_counter() - started) * 1000, 1),
                    search_calls=sum(o.attempts for o in outcomes),
                    failed_queries=sum(1 for o in outcomes if not o.ok),
                    raw_hits=sum(len(o.hits) for o in outcomes),
                    sources_returned=len(state["sources"]),
                    document_searches=len(document_outcomes),
                    document_chunks_retrieved=sum(len(o.chunks) for o in document_outcomes),
                    document_candidates_retrieved=sum(len(o.candidates) for o in document_outcomes),
                    document_candidates_rejected=sum(
                        max(0, len(o.candidates) - len(o.chunks)) for o in document_outcomes
                    ),
                    llm_calls=llm_calls,
                ),
                document_index=self._document_index(request.document_ids),
                document_retrieval=document_diagnostics,
                analysis=analysis,
                report=state.get("report"),
            )
            self.store.save(result)
            logger.info(
                "Research run finished: status=%s sources=%d duration_ms=%.1f",
                result.status.value,
                len(result.sources),
                result.metrics.duration_ms,
            )
            return result
        finally:
            run_id_var.reset(token)

    def get(self, run_id: UUID) -> ResearchResult | None:
        return self.store.get(run_id)
