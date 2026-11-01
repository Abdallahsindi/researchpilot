"""Lazy imports for optional dependencies, with install hints instead of bare ImportErrors."""

import importlib
from types import ModuleType

INSTALL_HINTS = {
    "faiss": "pip install faiss-cpu   (or: uv sync --group rag)",
    "sentence_transformers": "pip install sentence-transformers   (or: uv sync --group rag)",
    "torch": "pip install torch   (or: uv sync --group llm)",
    "transformers": "pip install transformers accelerate   (or: uv sync --group llm)",
    "psycopg": "pip install 'psycopg[binary]'   (or: uv sync --group supabase)",
}


class MissingDependencyError(ImportError):
    pass


def require(module: str, purpose: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = INSTALL_HINTS.get(module, f"pip install {module}")
        raise MissingDependencyError(
            f"The optional package '{module}' is required for {purpose}. Install it with: {hint}"
        ) from exc
