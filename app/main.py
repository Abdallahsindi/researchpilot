"""FastAPI entry point: `uvicorn app.main:app --reload`."""

# ruff: noqa: E402

import os

# FAISS and PyTorch both bring OpenMP runtimes. On macOS their worker pools can collide
# while the local model loads and crash the process with SIGSEGV inside libomp. Limit the
# pools before either library is imported; this must stay above the application imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collections.abc import Callable

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.rag.retrieval import DocumentLibrary
from app.services.documents import build_document_library
from app.services.llm import LLM
from app.services.research import ResearchService
from app.services.storage import JsonRunStore
from app.tools.web_search import SearchProvider, get_search_provider


def create_app(
    settings: Settings | None = None,
    provider: SearchProvider | None = None,
    library_factory: Callable[[], DocumentLibrary] | None = None,
    llm_factory: Callable[[], LLM] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="ResearchPilot", version="0.1.0")
    app.state.research_service = ResearchService(
        settings=settings,
        provider=provider or get_search_provider(settings),
        store=JsonRunStore(settings.data_dir / "runs"),
        # Called on the first document request, so startup never loads an embedding model.
        library_factory=library_factory or (lambda: build_document_library(settings)),
        # Called on the first analyze=true request; startup never loads the LLM.
        llm_factory=llm_factory,
    )
    app.include_router(router)
    return app


app = create_app()
