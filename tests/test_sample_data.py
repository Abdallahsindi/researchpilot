"""The synthetic test report in sample_data/pdfs/ and its companion question files."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import HashingEmbeddings
from app.rag.pdf import load_pdf
from app.rag.retrieval import VectorRetriever
from app.rag.vector_store import FaissVectorStore

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "sample_data" / "pdfs"
CASES = json.loads((PDF_DIR / "researchpilot_test_report_cases.json").read_text())


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_test_report", ROOT / "scripts" / "make_test_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def document():
    return load_pdf(PDF_DIR / CASES["pdf"])


def test_committed_files_match_the_generator():
    gen = _generator()
    pdf = gen.build_pdf(
        [text for _, text in gen.PAGES],
        page_titles=[title for title, _ in gen.PAGES],
        metadata={
            "Title": "Campus Energy Storage Pilot - Synthetic Test Report",
            "Author": "ResearchPilot test data",
        },
    )
    assert (PDF_DIR / CASES["pdf"]).read_bytes() == pdf  # regenerate after editing the script
    assert CASES["cases"] == gen.CASES
    assert (PDF_DIR / "researchpilot_test_questions.md").read_text() == (gen.questions_markdown())


def test_report_is_multi_page_and_labelled_synthetic(document):
    assert document.page_count == 5 and not document.warnings
    assert "SYNTHETIC TEST DATA" in document.pages[0].text
    assert "Table 1. Example battery specifications (synthetic test data)" in document.pages[1].text
    assert "Table 2. Installed cost (synthetic test data)" in document.pages[3].text


@pytest.mark.parametrize(
    "case", [c for c in CASES["cases"] if c["answerable"]], ids=lambda c: c["id"]
)
def test_expected_facts_are_on_the_expected_pages(document, case):
    for fact in case["expected_facts"]:
        pages = [p.page_number for p in document.pages if fact in " ".join(p.text.split())]
        assert set(pages) & set(case["expected_pages"]), (fact, pages)


def test_offline_retrieval_finds_expected_pages(document):
    embedder = HashingEmbeddings()
    store = FaissVectorStore(embedder.dimension)
    chunks = chunk_document(document)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = VectorRetriever(embedder, store)

    for case in CASES["cases"]:
        if case["answerable"]:
            pages = {r.chunk.page_number for r in retriever.retrieve(case["question"], top_k=3)}
            assert pages & set(case["expected_pages"]), (case["id"], pages)
