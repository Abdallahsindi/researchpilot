"""LangGraph research workflow.

    START -> plan -> search ×N           -> collect_sources -> assess_evidence
                   -> document_search ×N  ↗
    assess_evidence -> END                          (analyze=false: sources only)
                    -> report_writer                (analyze=true, evidence insufficient)
                    -> analyst -> verifier -> report_writer -> END
                       ^            |
                       +------------+  at most `analysis_max_revisions` revision rounds

Agents:
- Planner (`plan`): deterministic query planning from the question, criteria, and goals.
- Researcher (`search`, `document_search`, `collect_sources`, `assess_evidence`): parallel web
  and PDF retrieval, deduplication, ranking, and an evidence-sufficiency check.
- Analyst: the local LLM turns the collected sources into structured, cited findings.
- Verifier: code checks every citation, plus an optional LLM review per claim and source.
- Report Writer: deterministic Markdown from the verified results.
"""

import asyncio
import logging
from collections.abc import Callable
from time import perf_counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.agents.analyst import run_analyst
from app.agents.planner import plan_research
from app.agents.report_writer import render_report
from app.agents.verifier import ChunkLookup, llm_reviewer, verify_analysis
from app.config import Settings
from app.graph.state import DocumentSearchTask, ResearchState, SearchTask
from app.rag.models import RetrievalResult, RetrievedChunk
from app.schemas.analysis import AnalysisResult, AnalysisStatus, ReviewStatus, VerificationStatus
from app.schemas.research import (
    DocumentSearchOutcome,
    RunStatus,
    SearchHit,
    SearchOutcome,
    Source,
    SourceType,
    is_demo_url,
)
from app.services.llm import LLM
from app.tools.web_search import SearchProvider, SearchProviderError

logger = logging.getLogger(__name__)

# (query, top_k, document_ids) -> retrieval result (or accepted chunks for older callers)
DocumentRetriever = Callable[[str, int, list[str]], RetrievalResult | list[RetrievedChunk]]


def normalize_url(url: str) -> str:
    """Canonical form used to detect the same page returned by different queries."""
    parts = urlsplit(url.strip())
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    )
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.removeprefix("www.")


def _query_order(query_ids: list[str]) -> list[str]:
    return sorted(set(query_ids), key=lambda qid: int(qid[1:]))


def merge_document_outcomes(
    outcomes: list[DocumentSearchOutcome], max_sources: int
) -> list[tuple[RetrievedChunk, DocumentSearchOutcome, list[str]]]:
    """Merge document results, taking one passage per query before filling by score.

    Multi-part questions dispatch one scoped query per part. Ranking only by score can spend
    the whole document-source budget on the part with the strongest matches, so the first
    pass gives each planned query a representative passage. Remaining capacity is filled by
    score. Duplicate chunks keep the best score and all query IDs that surfaced them.
    """
    merged: dict[str, tuple[RetrievedChunk, DocumentSearchOutcome, list[str]]] = {}
    by_query: dict[str, list[str]] = {}
    for outcome in outcomes:
        for item in outcome.chunks:
            key = item.chunk.chunk_id
            if key not in merged:
                merged[key] = (item, outcome, [])
            best, _, query_ids = merged[key]
            if outcome.query.id not in query_ids:
                query_ids.append(outcome.query.id)
            if item.score > best.score:
                merged[key] = (item, outcome, query_ids)
            by_query.setdefault(outcome.query.id, []).append(key)

    for query_id, keys in by_query.items():
        by_query[query_id] = sorted(
            dict.fromkeys(keys), key=lambda key: merged[key][0].score, reverse=True
        )

    selected: list[tuple[RetrievedChunk, DocumentSearchOutcome, list[str]]] = []
    selected_keys: set[str] = set()
    for query_id in _query_order(list(by_query)):
        for key in by_query[query_id]:
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(merged[key])
                break
        if len(selected) == max_sources:
            return selected

    remaining = sorted(
        (value for key, value in merged.items() if key not in selected_keys),
        key=lambda value: value[0].score,
        reverse=True,
    )
    return (selected + remaining)[:max_sources]


def build_workflow(
    provider: SearchProvider,
    settings: Settings,
    document_retriever: DocumentRetriever | None = None,
    llm_provider: Callable[[], LLM] | None = None,
    chunk_lookup: ChunkLookup | None = None,
) -> CompiledStateGraph:
    """`llm_provider` returns the (lazily created) shared model; it is only called when a
    request has analyze=true, so non-analysis runs never touch it."""

    def plan(state: ResearchState) -> dict:
        research_plan = plan_research(state["request"], settings.max_search_queries)
        logger.info("Planned %d search queries", len(research_plan.queries))
        return {"plan": research_plan}

    def dispatch_searches(state: ResearchState) -> list[Send]:
        queries = state["plan"].queries
        sends = [Send("search", {"query": query}) for query in queries]
        document_ids = state["request"].document_ids
        if document_ids:
            sends += [
                Send("document_search", {"query": query, "document_ids": document_ids})
                for query in queries
            ]
        return sends

    async def search(task: SearchTask) -> dict:
        query = task["query"]
        max_attempts = settings.search_max_retries + 1
        started = perf_counter()
        hits: list[SearchHit] = []
        error: str | None = None
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            retryable = False
            try:
                hits = await asyncio.wait_for(
                    provider.search(query.text, settings.max_results_per_query),
                    timeout=settings.search_timeout_seconds,
                )
                error = None
                break
            except SearchProviderError as exc:
                error, retryable = str(exc), exc.retryable
            except TimeoutError:
                error = f"Search timed out after {settings.search_timeout_seconds}s"
                retryable = True
            except Exception as exc:  # one broken query must not abort the whole run
                logger.exception("Unexpected error searching %s", query.id)
                error = f"Unexpected search error: {exc}"

            logger.warning("Search %s attempt %d failed: %s", query.id, attempts, error)
            if not retryable or attempts == max_attempts:
                break
            await asyncio.sleep(settings.search_retry_backoff_seconds * attempts)

        outcome = SearchOutcome(
            query=query,
            provider=provider.name,
            hits=hits,
            error=error,
            attempts=attempts,
            duration_ms=round((perf_counter() - started) * 1000, 1),
        )
        logger.info("Search %s returned %d hits", query.id, len(hits))
        return {"search_outcomes": [outcome]}

    async def document_search(task: DocumentSearchTask) -> dict:
        query = task["query"]
        started = perf_counter()
        retrieval = RetrievalResult()
        error: str | None = None
        if document_retriever is None:
            error = "Document retrieval is not configured"
        else:
            try:
                # Embedding + vector search is CPU-bound; keep it off the event loop.
                result = await asyncio.to_thread(
                    document_retriever, query.text, settings.document_top_k, task["document_ids"]
                )
                if isinstance(result, RetrievalResult):
                    retrieval = result
                else:
                    # Backwards-compatible with simple retriever callbacks in tests/callers.
                    retrieval = RetrievalResult(
                        chunks=list(result),
                        candidates=list(result),
                        min_score=settings.document_min_score,
                    )
            except Exception as exc:  # a failing document search must not abort the run
                logger.exception("Document search failed for %s", query.id)
                error = f"Document search error: {exc}"
        outcome = DocumentSearchOutcome(
            query=query,
            chunks=retrieval.chunks,
            candidates=retrieval.candidates,
            indexed_chunks=retrieval.indexed_chunks,
            selected_chunks=retrieval.selected_chunks,
            selected_document_ids=retrieval.selected_document_ids,
            min_score=retrieval.min_score,
            fallback_used=retrieval.fallback_used,
            rejection_reason=retrieval.rejection_reason,
            error=error,
            duration_ms=round((perf_counter() - started) * 1000, 1),
        )
        logger.info(
            "Document search %s returned %d/%d candidates above the threshold",
            query.id,
            len(retrieval.chunks),
            len(retrieval.candidates),
        )
        return {"document_outcomes": [outcome]}

    def collect_document_sources(state: ResearchState) -> list[dict]:
        ranked = merge_document_outcomes(
            state.get("document_outcomes", []), settings.max_document_sources
        )
        return [
            {
                "source_type": SourceType.DOCUMENT,
                "title": f"{item.chunk.filename}, page {item.chunk.page_number}",
                "snippet": item.chunk.text,
                "provider": "documents",
                "score": round(item.score, 4),
                "retrieved_at": outcome.retrieved_at,
                "query_ids": _query_order(query_ids),
                "document_id": item.chunk.document_id,
                "filename": item.chunk.filename,
                "page_number": item.chunk.page_number,
                "chunk_id": item.chunk.chunk_id,
            }
            for item, outcome, query_ids in ranked
        ]

    def collect_sources(state: ResearchState) -> dict:
        merged: dict[str, tuple[SearchHit, SearchOutcome, list[str]]] = {}
        for outcome in state["search_outcomes"]:
            for hit in outcome.hits:
                key = normalize_url(hit.url)
                if key not in merged:
                    merged[key] = (hit, outcome, [outcome.query.id])
                    continue
                best, _, query_ids = merged[key]
                query_ids.append(outcome.query.id)
                if (hit.score or 0) > (best.score or 0):
                    merged[key] = (hit, outcome, query_ids)

        # Rank by provider score, then by how many queries surfaced the source.
        ranked = sorted(
            merged.values(), key=lambda item: (item[0].score or 0, len(item[2])), reverse=True
        )
        web_sources = [
            {
                # Demo-corpus entries are synthetic even if a live provider ever returned one.
                "source_type": SourceType.DEMO
                if outcome.provider == "demo" or is_demo_url(hit.url)
                else SourceType.WEB,
                "url": hit.url,
                "domain": _domain(hit.url),
                "title": hit.title,
                "snippet": hit.snippet,
                "provider": outcome.provider,
                "score": hit.score,
                "published_date": hit.published_date,
                "retrieved_at": outcome.retrieved_at,
                "query_ids": _query_order(query_ids),
            }
            for hit, outcome, query_ids in ranked[: settings.max_sources]
        ]
        # The user's own documents come first, then web results.
        combined = collect_document_sources(state) + web_sources
        sources = [Source(id=f"S{i}", **fields) for i, fields in enumerate(combined, start=1)]
        logger.info("Collected %d unique sources", len(sources))
        return {"sources": sources}

    def assess_evidence(state: ResearchState) -> dict:
        outcomes = state["search_outcomes"]
        document_outcomes = state.get("document_outcomes", [])
        sources = state["sources"]
        failed = [o for o in outcomes if not o.ok]
        failed_documents = [o for o in document_outcomes if not o.ok]
        warnings: list[str] = []

        total = len(outcomes) + len(document_outcomes)
        if len(failed) + len(failed_documents) == total:
            status = RunStatus.FAILED
            warnings.append(f"All {total} searches failed; no evidence was collected.")
        elif len(sources) < settings.min_sources:
            status = RunStatus.INSUFFICIENT_EVIDENCE
            warnings.append(
                f"Found {len(sources)} relevant source(s); at least {settings.min_sources} "
                "are required for a supported answer."
            )
        else:
            status = RunStatus.COMPLETED

        if failed and status is not RunStatus.FAILED:
            warnings.append(
                f"{len(failed)} of {len(outcomes)} searches failed; results may be incomplete."
            )
        if failed_documents and status is not RunStatus.FAILED:
            warnings.append(
                f"{len(failed_documents)} of {len(document_outcomes)} document searches failed."
            )
        if document_outcomes and not any(s.source_type is SourceType.DOCUMENT for s in sources):
            completed = [o for o in document_outcomes if o.ok]
            best = max(
                (candidate for outcome in completed for candidate in outcome.candidates),
                key=lambda candidate: candidate.score,
                default=None,
            )
            threshold = completed[0].min_score if completed else settings.document_min_score
            if best is None:
                warnings.append(
                    "No indexed passage was returned for the selected documents "
                    f"({len(state['request'].document_ids)} selected)."
                )
            else:
                warnings.append(
                    "No passage in the selected documents met the similarity threshold "
                    f"({threshold:.2f}); the best was {best.chunk.filename}, page "
                    f"{best.chunk.page_number} at {best.score:.2f}."
                )
        if sources and all(s.source_type is SourceType.DEMO for s in sources):
            warnings.append(
                "All sources are synthetic Demo-mode entries (demo.example), not real evidence."
            )

        logger.info("Evidence assessment: %s", status.value)
        return {"status": status, "warnings": warnings}

    def route_after_assessment(state: ResearchState) -> str:
        if not state["request"].analyze:
            return END
        if state["status"] is not RunStatus.COMPLETED:
            return "report_writer"
        return "analyst"

    async def get_llm() -> LLM:
        if llm_provider is None:
            raise RuntimeError("No language model is configured for analysis.")
        # Creating and loading the model can take minutes on first use (download).
        llm = await asyncio.wait_for(
            asyncio.to_thread(llm_provider), timeout=settings.llm_load_timeout_seconds
        )
        load = getattr(llm, "load", None)
        if load is not None:
            await asyncio.wait_for(
                asyncio.to_thread(load), timeout=settings.llm_load_timeout_seconds
            )
        return llm

    def analysis_sources(state: ResearchState) -> list[Source]:
        return state["sources"][: settings.analysis_max_sources]

    async def analyst(state: ResearchState) -> dict:
        analysis = state.get("analysis") or AnalysisResult(status=AnalysisStatus.FAILED)
        analysis = analysis.model_copy(deep=True)
        attempt_number = len(analysis.attempts) + 1
        try:
            llm = await get_llm()
            analysis.model_id = llm.model_id
            attempt = await asyncio.wait_for(
                asyncio.to_thread(
                    run_analyst,
                    llm,
                    state["request"],
                    analysis_sources(state),
                    attempt=attempt_number,
                    max_chars=settings.analysis_source_chars,
                    max_new_tokens=settings.analysis_max_new_tokens,
                    feedback=state.get("revision_feedback"),
                ),
                timeout=settings.analysis_timeout_seconds,
            )
        except TimeoutError:
            analysis.errors.append(
                f"Analyst attempt {attempt_number} timed out after "
                f"{settings.analysis_timeout_seconds:.0f}s."
            )
            return {"analysis": analysis, "analyst_failed": True}
        except Exception as exc:  # model unavailable, failed to load, or inference error
            logger.exception("Analyst attempt %d failed", attempt_number)
            analysis.errors.append(f"Analyst model error: {type(exc).__name__}: {exc}")
            return {"analysis": analysis, "analyst_failed": True}

        analysis.attempts.append(attempt)
        metrics = analysis.metrics
        metrics.analyst_calls += 1
        metrics.llm_ms += attempt.duration_ms
        metrics.prompt_tokens += attempt.prompt_tokens
        metrics.completion_tokens += attempt.completion_tokens
        update: dict = {"analysis": analysis, "analyst_failed": False}
        if attempt.parse_error:
            analysis.errors.append(f"Attempt {attempt_number}: {attempt.parse_error}")
            update["revision_feedback"] = (
                f"{attempt.parse_error} Reply with a single JSON object in the required shape."
            )
        logger.info(
            "Analyst attempt %d: %s",
            attempt_number,
            attempt.parse_error or f"{len(attempt.output.findings)} findings",
        )
        return update

    def revisions_left(state: ResearchState) -> bool:
        return len(state["analysis"].attempts) <= settings.analysis_max_revisions

    def route_after_analyst(state: ResearchState) -> str:
        if state.get("analyst_failed"):
            return "report_writer"  # model error or timeout: do not retry inference
        last = state["analysis"].attempts[-1]
        if last.parse_error:
            return "analyst" if revisions_left(state) else "report_writer"
        if last.output.insufficient_evidence:
            return "report_writer"
        return "verifier"

    async def verifier(state: ResearchState) -> dict:
        analysis = state["analysis"].model_copy(deep=True)
        attempt = analysis.attempts[-1]
        llm = None
        review_status = ReviewStatus.NOT_PERFORMED
        review_reason = "LLM review is disabled for this run"
        if settings.verifier_llm_review:
            review_reason = "LLM review is unavailable"
            try:
                llm = await get_llm()
            except Exception as exc:
                review_reason = f"LLM review is unavailable: {exc}"
                analysis.errors.append(f"LLM review unavailable: {exc}")

        review_calls = [0]
        base_reviewer = llm_reviewer(llm, settings.analysis_source_chars) if llm else None

        def counted_reviewer(claim, source):
            review_calls[0] += 1
            return base_reviewer(claim, source)

        def verify(reviewer, *, status=review_status, reason=review_reason):
            return verify_analysis(
                attempt.output,
                analysis_sources(state),
                criteria=state["request"].criteria,
                question=state["request"].question,
                invented_ids=attempt.invented_source_ids,
                chunk_lookup=chunk_lookup,
                reviewer=reviewer,
                max_reviews=settings.verifier_max_reviews,
                review_unavailable_status=status,
                review_unavailable_reason=reason,
            )

        try:
            report = await asyncio.wait_for(
                asyncio.to_thread(verify, counted_reviewer if base_reviewer else None),
                timeout=settings.analysis_timeout_seconds,
            )
        except TimeoutError:
            # Code checks alone are fast; claims simply stay unverified without the review.
            analysis.errors.append("LLM review timed out; verification used code checks only.")
            report = await asyncio.to_thread(
                verify,
                None,
                status=ReviewStatus.FAILED,
                reason="the LLM review timed out",
            )
        analysis.metrics.review_calls += review_calls[0]
        analysis.verification = report

        update: dict = {"analysis": analysis, "verification": report, "revision_feedback": None}
        if report.needs_revision and revisions_left({**state, "analysis": analysis}):
            update["revision_feedback"] = report.feedback
        logger.info(
            "Verifier: %d verified, %d unverified, %d rejected",
            report.count(VerificationStatus.VERIFIED),
            report.count(VerificationStatus.UNVERIFIED),
            len(report.claims)
            - report.count(VerificationStatus.VERIFIED)
            - report.count(VerificationStatus.UNVERIFIED),
        )
        return update

    def route_after_verifier(state: ResearchState) -> str:
        return "analyst" if state.get("revision_feedback") else "report_writer"

    def report_writer(state: ResearchState) -> dict:
        analysis = state.get("analysis")
        if analysis is None:  # routed here because the evidence was insufficient
            analysis = AnalysisResult(
                status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
                errors=[f"Analysis skipped: evidence assessment was '{state['status'].value}'."],
            )
        else:
            analysis = analysis.model_copy(deep=True)
            output = analysis.final_output
            report = analysis.verification
            if output is None:
                analysis.status = AnalysisStatus.FAILED
            elif output.insufficient_evidence:
                analysis.status = AnalysisStatus.INSUFFICIENT_EVIDENCE
            elif report and report.count(VerificationStatus.VERIFIED):
                analysis.status = AnalysisStatus.COMPLETED
            elif report and report.claims and not any(c.addresses_question for c in report.claims):
                # Every claim was off-topic: the sources do not answer this question.
                analysis.status = AnalysisStatus.INSUFFICIENT_EVIDENCE
            else:
                analysis.status = AnalysisStatus.NO_VERIFIED_FINDINGS
        markdown = render_report(
            state["request"],
            state["sources"],
            analysis,
            run_warnings=state.get("warnings", []),
            demo_mode=settings.demo_mode,
        )
        logger.info("Report written: analysis %s", analysis.status.value)
        return {"analysis": analysis, "report": markdown}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("document_search", document_search)
    graph.add_node("collect_sources", collect_sources)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("analyst", analyst)
    graph.add_node("verifier", verifier)
    graph.add_node("report_writer", report_writer)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch_searches, ["search", "document_search"])
    graph.add_edge("search", "collect_sources")
    graph.add_edge("document_search", "collect_sources")
    graph.add_edge("collect_sources", "assess_evidence")
    graph.add_conditional_edges(
        "assess_evidence", route_after_assessment, ["analyst", "report_writer", END]
    )
    graph.add_conditional_edges(
        "analyst", route_after_analyst, ["analyst", "verifier", "report_writer"]
    )
    graph.add_conditional_edges("verifier", route_after_verifier, ["analyst", "report_writer"])
    graph.add_edge("report_writer", END)
    return graph.compile()
