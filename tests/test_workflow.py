import asyncio

from app.agents.planner import plan_research
from app.graph.workflow import normalize_url
from app.schemas.research import ResearchRequest, RunStatus, Source, SourceType
from app.tools.web_search import SearchProviderError
from tests.conftest import StubProvider, hit

VECTOR_DB_REQUEST = ResearchRequest(
    question="Should a small team use pgvector or a dedicated vector database?",
    criteria=["cost", "latency"],
)


async def test_demo_run_completes_with_cited_sources(make_service):
    service = make_service()

    result = await service.run(VECTOR_DB_REQUEST)

    assert result.status is RunStatus.COMPLETED
    assert result.mode == "demo"
    assert [q.id for q in result.plan.queries] == ["Q1", "Q2", "Q3"]
    assert len(result.sources) >= 2
    assert [s.id for s in result.sources] == [f"S{i}" for i in range(1, len(result.sources) + 1)]
    assert all(s.url.startswith("https://demo.example/") for s in result.sources)
    assert all(s.query_ids and s.provider == "demo" for s in result.sources)
    assert result.errors == []
    assert result.metrics.search_calls == 3
    assert result.metrics.sources_returned == len(result.sources)
    assert service.get(result.run_id) == result


async def test_unrelated_question_reports_insufficient_evidence(make_service):
    service = make_service()

    result = await service.run(
        ResearchRequest(question="What is the best recipe for a sourdough bread starter?")
    )

    assert result.status is RunStatus.INSUFFICIENT_EVIDENCE
    assert result.sources == []
    assert result.errors == []
    assert any("at least 2" in w for w in result.warnings)


async def test_too_few_sources_is_insufficient_evidence(make_service):
    provider = StubProvider(lambda q, n: [hit("https://a.example/only")])
    service = make_service(provider, min_sources=3)

    result = await service.run(VECTOR_DB_REQUEST)

    assert result.status is RunStatus.INSUFFICIENT_EVIDENCE
    assert len(result.sources) == 1


async def test_all_searches_failing_marks_run_failed(make_service):
    provider = StubProvider(lambda q, n: SearchProviderError("HTTP 401", retryable=False))
    service = make_service(provider)

    result = await service.run(VECTOR_DB_REQUEST)

    assert result.status is RunStatus.FAILED
    assert result.sources == []
    assert len(result.errors) == 3
    assert all(e.message == "HTTP 401" and e.attempts == 1 for e in result.errors)
    assert result.metrics.failed_queries == 3
    assert service.get(result.run_id) is not None


async def test_partial_failure_completes_with_warning(make_service):
    def handler(query, n):
        if query.endswith("cost"):
            return SearchProviderError("HTTP 500", retryable=True)
        return [hit(f"https://site.example/{query.split()[-1]}"), hit("https://other.example/x")]

    service = make_service(StubProvider(handler), search_max_retries=1)

    result = await service.run(VECTOR_DB_REQUEST)

    assert result.status is RunStatus.COMPLETED
    assert [e.query_id for e in result.errors] == ["Q2"]
    assert result.errors[0].attempts == 2
    assert any("1 of 3 searches failed" in w for w in result.warnings)
    assert result.metrics.search_calls == 4


async def test_transient_failure_is_retried(make_service):
    def handler(query, n):
        if n == 1:
            return SearchProviderError("HTTP 429", retryable=True)
        return [hit("https://a.example/1"), hit("https://b.example/2")]

    provider = StubProvider(handler)
    service = make_service(provider, search_max_retries=2)

    result = await service.run(ResearchRequest(question="A question with no criteria at all"))

    assert result.status is RunStatus.COMPLETED
    assert result.errors == []
    assert result.metrics.search_calls == 2
    assert len(provider.calls) == 2


async def test_non_retryable_failure_is_not_retried(make_service):
    provider = StubProvider(lambda q, n: SearchProviderError("bad key", retryable=False))
    service = make_service(provider, search_max_retries=3)

    await service.run(ResearchRequest(question="A question with no criteria at all"))

    assert len(provider.calls) == 1


async def test_unexpected_exception_is_recorded_not_raised(make_service):
    provider = StubProvider(lambda q, n: KeyError("results"))
    service = make_service(provider, search_max_retries=2)

    result = await service.run(ResearchRequest(question="A question with no criteria at all"))

    assert result.status is RunStatus.FAILED
    assert "Unexpected search error" in result.errors[0].message
    assert len(provider.calls) == 1


async def test_slow_search_times_out(make_service):
    async def slow(query, n):
        await asyncio.sleep(1)
        return [hit("https://a.example/late")]

    service = make_service(StubProvider(slow), search_timeout_seconds=0.05, search_max_retries=0)

    result = await service.run(ResearchRequest(question="A question with no criteria at all"))

    assert result.status is RunStatus.FAILED
    assert "timed out" in result.errors[0].message


async def test_duplicate_urls_are_merged_across_queries(make_service):
    def handler(query, n):
        if query.endswith("latency"):
            return [hit("https://Docs.example/page/?utm_source=x", score=0.9)]
        return [hit("https://docs.example/page", score=0.5), hit("https://b.example/2", 0.7)]

    service = make_service(StubProvider(handler))

    result = await service.run(VECTOR_DB_REQUEST)

    assert len(result.sources) == 2
    top = result.sources[0]
    assert top.id == "S1"
    assert top.score == 0.9
    assert top.query_ids == ["Q1", "Q2", "Q3"]
    assert result.metrics.raw_hits == 5


async def test_sources_are_capped_at_max_sources(make_service):
    provider = StubProvider(
        lambda q, n: [hit(f"https://x.example/{n}-{i}", score=i / 10) for i in range(5)]
    )
    service = make_service(provider, max_sources=4)

    result = await service.run(VECTOR_DB_REQUEST)

    assert len(result.sources) == 4
    assert result.sources[0].score >= result.sources[-1].score


def test_planner_builds_queries_from_criteria_and_goals():
    request = ResearchRequest(
        question="Which LLM provider fits our app?",
        criteria=["cost", "latency", "cost"],
        goals=["low ops burden"],
    )

    plan = plan_research(request, max_queries=10)

    assert [q.text for q in plan.queries] == [
        "Which LLM provider fits our app?",
        "Which LLM provider fits our app? cost",
        "Which LLM provider fits our app? latency",
        "Which LLM provider fits our app? low ops burden",
    ]
    assert plan_research(request, max_queries=2).queries[-1].id == "Q2"


def test_normalize_url():
    assert normalize_url("HTTPS://Example.com/a/?utm_medium=x&id=3#frag") == (
        "https://example.com/a?id=3"
    )


async def test_demo_sources_are_labelled_synthetic(make_service):
    result = await make_service().run(VECTOR_DB_REQUEST)

    assert {s.source_type for s in result.sources} == {SourceType.DEMO}
    assert any("not real evidence" in w for w in result.warnings)


async def test_live_provider_returning_demo_url_is_still_demo(make_service):
    provider = StubProvider(
        lambda q, n: [hit("https://demo.example/fake"), hit("https://real.example/page")]
    )

    result = await make_service(provider).run(VECTOR_DB_REQUEST)

    types = {s.url: s.source_type for s in result.sources}
    assert types == {
        "https://demo.example/fake": SourceType.DEMO,
        "https://real.example/page": SourceType.WEB,
    }


async def test_document_search_failure_is_recorded(make_service, sample_pdf):
    service = make_service()
    info, _ = service.documents().ingest(sample_pdf, "report.pdf")

    def broken(*args, **kwargs):
        raise RuntimeError("index unavailable")

    service.documents().retriever.retrieve_with_diagnostics = broken
    result = await service.run(
        ResearchRequest(
            question="Should a small team use pgvector?", document_ids=[info.document_id]
        )
    )

    assert result.status is RunStatus.COMPLETED  # web evidence still collected
    assert any("document searches failed" in w for w in result.warnings)
    assert any("index unavailable" in e.message for e in result.errors)


def test_legacy_stored_sources_get_inferred_types():
    legacy = {
        "id": "S1",
        "url": "https://demo.example/x",
        "domain": "demo.example",
        "title": "t",
        "snippet": "s",
        "provider": "tavily",
        "retrieved_at": "2026-10-22T00:00:00Z",
        "query_ids": ["Q1"],
    }

    assert Source.model_validate(legacy).source_type is SourceType.DEMO
    real = {**legacy, "url": "https://example.org/a"}
    assert Source.model_validate(real).source_type is SourceType.WEB
