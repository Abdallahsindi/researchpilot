import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.tools.web_search import (
    DemoSearchProvider,
    SearchProviderError,
    TavilySearchProvider,
    get_search_provider,
)


def tavily_with(handler) -> TavilySearchProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TavilySearchProvider("tvly-test", client=client)


async def test_tavily_maps_results_and_sends_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a.com", "title": "A", "content": "text", "score": 0.91},
                    {"url": "https://b.com", "title": "", "content": "more"},
                    {"title": "no url"},
                ]
            },
        )

    hits = await tavily_with(handler).search("pgvector", max_results=3)

    assert seen["auth"] == "Bearer tvly-test"
    assert b'"max_results":3' in seen["body"].replace(b" ", b"")
    assert [(h.url, h.title, h.snippet, h.score) for h in hits] == [
        ("https://a.com", "A", "text", 0.91),
        ("https://b.com", "https://b.com", "more", None),
    ]


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (401, False)])
async def test_tavily_http_errors(status, retryable):
    provider = tavily_with(lambda request: httpx.Response(status))

    with pytest.raises(SearchProviderError) as exc_info:
        await provider.search("q", max_results=1)

    assert exc_info.value.retryable is retryable


async def test_tavily_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SearchProviderError) as exc_info:
        await tavily_with(handler).search("q", max_results=1)

    assert exc_info.value.retryable is True


async def test_demo_provider_ranks_relevant_documents():
    hits = await DemoSearchProvider().search("pgvector postgres vector database", max_results=3)

    assert hits
    assert "pgvector" in hits[0].title
    assert len(hits) <= 3
    assert await DemoSearchProvider().search("sourdough bread", max_results=3) == []


def test_live_mode_requires_tavily_key():
    with pytest.raises(ValidationError, match="TAVILY_API_KEY"):
        Settings(_env_file=None, demo_mode=False, tavily_api_key=None)


def test_provider_factory_selects_by_mode():
    demo = Settings(_env_file=None, demo_mode=True)
    live = Settings(_env_file=None, demo_mode=False, tavily_api_key="tvly-x")

    assert get_search_provider(demo).name == "demo"
    assert get_search_provider(live).name == "tavily"
