"""Web search providers behind a common interface.

The workflow only depends on `SearchProvider`, so adding a provider means adding a class here
and a branch in `get_search_provider`.
"""

import json
import logging
import re
from functools import cached_property
from importlib import resources
from typing import Protocol

import httpx

from app.config import Settings
from app.schemas.research import SearchHit

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class TavilySearchProvider:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self, api_key: str, *, timeout: float = 20.0, client: httpx.AsyncClient | None = None
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        payload = {"query": query, "max_results": max_results, "search_depth": "basic"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            if self._client is not None:
                response = await self._client.post(
                    self.endpoint, json=payload, headers=headers, timeout=self._timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self.endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise SearchProviderError("Tavily request timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise SearchProviderError(f"Tavily connection error: {exc}", retryable=True) from exc

        if response.status_code != 200:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise SearchProviderError(
                f"Tavily returned HTTP {response.status_code}", retryable=retryable
            )

        try:
            results = response.json().get("results", [])
        except ValueError as exc:
            raise SearchProviderError("Tavily returned invalid JSON", retryable=False) from exc

        return [
            SearchHit(
                url=item["url"],
                title=item.get("title") or item["url"],
                snippet=item.get("content", ""),
                score=item.get("score"),
                published_date=item.get("published_date"),
            )
            for item in results
            if item.get("url")
        ]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be best by can do does for from how i in is it of on or our should "
    "the their this to us vs versus we what when which who why with".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


class DemoSearchProvider:
    """Offline keyword search over a small bundled corpus.

    Results point at `demo.example` URLs so they can never be mistaken for real citations.
    Queries unrelated to the corpus return nothing, which exercises the missing-evidence path.
    """

    name = "demo"
    min_score = 0.25

    @cached_property
    def corpus(self) -> list[dict]:
        text = resources.files("app.tools").joinpath("demo_corpus.json").read_text("utf-8")
        return json.loads(text)

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored = []
        for doc in self.corpus:
            doc_tokens = _tokens(f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}")
            score = len(query_tokens & doc_tokens) / len(query_tokens)
            if score >= self.min_score:
                scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchHit(
                url=doc["url"],
                title=doc["title"],
                snippet=doc["content"],
                score=round(score, 3),
                published_date=doc.get("published_date"),
            )
            for score, doc in scored[:max_results]
        ]


def get_search_provider(settings: Settings) -> SearchProvider:
    if settings.demo_mode:
        return DemoSearchProvider()
    if settings.search_provider == "tavily":
        assert settings.tavily_api_key is not None  # enforced by Settings validation
        return TavilySearchProvider(
            settings.tavily_api_key.get_secret_value(), timeout=settings.search_timeout_seconds
        )
    raise ValueError(f"Unknown search provider: {settings.search_provider}")
