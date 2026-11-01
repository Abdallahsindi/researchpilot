import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings
from app.rag.embeddings import HashingEmbeddings
from app.rag.retrieval import DocumentLibrary
from app.schemas.research import SearchHit
from app.services.documents import build_document_library
from app.services.llm import Generation
from app.services.research import ResearchService
from app.services.storage import JsonRunStore
from app.tools.web_search import DemoSearchProvider, SearchProvider
from tests.pdf_factory import build_pdf


class StubProvider:
    """Search provider driven by a handler `(query, call_number) -> hits | Exception`."""

    name = "stub"

    def __init__(self, handler: Callable) -> None:
        self.handler = handler
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.calls.append(query)
        result = self.handler(query, len(self.calls))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Exception):
            raise result
        return result


def hit(url: str, score: float = 0.8, title: str = "Title") -> SearchHit:
    return SearchHit(url=url, title=title, snippet=f"Snippet for {url}", score=score)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def factory(**overrides) -> Settings:
        values = {
            "app_env": "test",
            "data_dir": tmp_path,
            "demo_mode": True,
            "search_retry_backoff_seconds": 0,
            # Lexical embeddings keep tests offline; no model download.
            "embedding_provider": "hashing",
            "document_min_score": 0.1,
            **overrides,
        }
        return Settings(_env_file=None, **values)

    return factory


@pytest.fixture
def make_service(make_settings) -> Callable[..., ResearchService]:
    def factory(provider: SearchProvider | None = None, **overrides) -> ResearchService:
        settings = make_settings(**overrides)
        return ResearchService(
            settings=settings,
            provider=provider or DemoSearchProvider(),
            store=JsonRunStore(settings.data_dir / "runs"),
            library_factory=lambda: build_document_library(settings),
        )

    return factory


SAMPLE_PAGES = [
    "Solar array overview. The rooftop solar array at the north campus has a peak capacity "
    "of 1.2 megawatts and powers three buildings.",
    "Battery storage. The battery system uses lithium iron phosphate cells with a usable "
    "capacity of 2.4 megawatt hours. Lithium iron phosphate was chosen for safety.",
    None,  # no text layer, like a scanned page
    "Project costs. The total installed cost was 3.1 million dollars and the payback period "
    "is about nine years.",
]


@pytest.fixture
def sample_pdf() -> bytes:
    return build_pdf(SAMPLE_PAGES, metadata={"Title": "Microgrid test report"})


@pytest.fixture
def library(tmp_path: Path) -> DocumentLibrary:
    return DocumentLibrary(tmp_path / "documents", HashingEmbeddings(), min_score=0.1)


class FakeLLM:
    """Scripted stand-in for the local model. It proves nothing about real model quality."""

    model_id = "fake-llm"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, *, max_new_tokens: int = 256) -> Generation:
        self.calls.append(messages)
        return Generation(self.output, prompt_tokens=100, completion_tokens=20, duration_ms=5.0)


class ScriptedLLM:
    """Scripted stand-in for the analysis model; tests the workflow, not model quality.

    `analyst` is a list of replies (the last one repeats) or a function of the user prompt.
    `review` is the reply to Verifier review prompts, or a function (claim, source_text).
    """

    model_id = "scripted-llm"

    def __init__(self, analyst, review="SUPPORTED", *, delay=0.0, error=None, review_delay=0.0):
        self.analyst = analyst
        self.review = review
        self.delay = delay
        self.review_delay = review_delay
        self.error = error
        self.analyst_calls: list[str] = []
        self.review_calls = 0
        self.loaded = False

    def load(self):
        self.loaded = True

    def generate(self, messages, *, max_new_tokens: int = 256) -> Generation:
        import time

        system, user = messages[0]["content"], messages[-1]["content"]
        if "fact checker" in system:
            self.review_calls += 1
            time.sleep(self.review_delay)
            claim = user.split("CLAIM: ", 1)[1].split("\n", 1)[0]
            text = user.split("<source>\n", 1)[1].split("\n</source>", 1)[0]
            reply = self.review(claim, text) if callable(self.review) else self.review
            return Generation(reply, prompt_tokens=50, completion_tokens=2, duration_ms=1.0)
        if self.error:
            raise self.error
        time.sleep(self.delay)
        self.analyst_calls.append(user)
        if callable(self.analyst):
            reply = self.analyst(user)
        else:
            reply = self.analyst[min(len(self.analyst_calls), len(self.analyst)) - 1]
        return Generation(reply, prompt_tokens=400, completion_tokens=80, duration_ms=10.0)


def source_ids_containing(prompt: str, phrase: str) -> list[str]:
    """IDs of the <source> blocks in an Analyst prompt whose text contains `phrase`."""
    import re

    blocks = re.findall(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.DOTALL)
    return [sid for sid, text in blocks if phrase.lower() in text.lower()]
