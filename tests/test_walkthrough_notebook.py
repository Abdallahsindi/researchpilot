"""Static checks on the local walkthrough notebook. They do not execute it or load a model."""

import ast
import importlib
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "researchpilot_project_walkthrough.ipynb"
)


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code(notebook):
    return [c for c in notebook.cells if c.cell_type == "code"]


def test_valid_and_compiles(notebook):
    nbformat.validate(notebook)
    for index, cell in enumerate(_code(notebook)):
        compile(cell.source, f"<cell {index}>", "exec")


def test_committed_without_outputs_or_secrets(notebook):
    assert all(c.outputs == [] and c.execution_count is None for c in _code(notebook))
    text = NOTEBOOK.read_text()
    for pattern in (r"hf_[A-Za-z0-9]{30,}", r"tvly-[A-Za-z0-9]{10,}", r"postgres(ql)?://\S+:\S+@"):
        assert not re.search(pattern, text)


def test_runs_locally_without_kaggle_paths_or_interactive_input(notebook):
    source = "\n".join(c.source for c in _code(notebook))
    assert "/kaggle/" not in source and "input(" not in source
    assert "/Users/" not in source  # no machine-specific paths


def test_imports_existing_project_code(notebook):
    for cell in _code(notebook):
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"


def test_sections_present_in_order(notebook):
    headings = [
        int(m.group(1))
        for c in notebook.cells
        if c.cell_type == "markdown"
        for line in c.source.splitlines()
        if (m := re.match(r"## (\d+)\.", line))
    ]
    assert headings == list(range(1, 13))
