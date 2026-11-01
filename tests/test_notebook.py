"""Static checks on the Kaggle notebook. They do not execute it or load any model."""

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "researchpilot_kaggle_demo.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _python_source(cell) -> str:
    """Cell source without IPython magics or shell escapes, which are not Python syntax."""
    lines = cell.source.splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))


def test_notebook_is_valid_nbformat(notebook):
    nbformat.validate(notebook)


def test_code_cells_compile(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        compile(_python_source(cell), f"<cell {index}>", "exec")


def test_sections_appear_in_order(notebook):
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    numbers = [int(m.group(1)) for h in headings if (m := re.match(r"## (\d+)\.", h))]
    assert numbers == list(range(1, 10))


def test_committed_notebook_has_no_outputs_or_secrets(notebook):
    for cell in _code_cells(notebook):
        assert cell.outputs == [] and cell.execution_count is None
    text = NOTEBOOK.read_text()
    for pattern in (r"hf_[A-Za-z0-9]{30,}", r"tvly-[A-Za-z0-9]{10,}", r"postgres(ql)?://\S+:\S+@"):
        assert not re.search(pattern, text), pattern


def test_imported_project_names_exist(notebook):
    for cell in _code_cells(notebook):
        tree = ast.parse(_python_source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"


def test_notebook_modules_import_without_api_dependencies():
    # Kaggle images may lack FastAPI, LangGraph and pydantic-settings; the notebook must not
    # need them. Block them and import everything the notebook uses.
    code = (
        "import sys\n"
        "for name in ('fastapi', 'langgraph', 'pydantic_settings', 'uvicorn'):\n"
        "    sys.modules[name] = None\n"
        "import app.rag.pdf, app.rag.chunking, app.rag.embeddings, app.rag.vector_store\n"
        "import app.rag.retrieval, app.rag.qa, app.rag.evaluation, app.services.llm\n"
        "heavy = [m for m in ('torch', 'transformers', 'sentence_transformers', 'faiss')"
        " if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
