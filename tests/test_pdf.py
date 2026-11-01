import io
import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from app.rag.chunking import chunk_document, split_text
from app.rag.pdf import (
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    load_pdf,
    parse_pdf,
)
from tests.pdf_factory import build_pdf

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data"


def test_extracts_text_per_page_with_metadata(sample_pdf):
    document = parse_pdf(sample_pdf, "report.pdf")

    assert document.filename == "report.pdf"
    assert document.page_count == 4
    assert [p.page_number for p in document.pages] == [1, 2, 3, 4]
    assert "1.2 megawatts" in document.pages[0].text
    assert "lithium iron phosphate" in document.pages[1].text
    assert document.pages[2].text == ""
    assert document.pages_with_text == 3
    assert document.metadata == {"title": "Microgrid test report"}
    assert document.warnings and "Page(s) 3" in document.warnings[0]
    assert "OCR is not supported" in document.warnings[0]


def test_document_id_is_stable_and_content_based(sample_pdf):
    first = parse_pdf(sample_pdf, "a.pdf")
    second = parse_pdf(sample_pdf, "renamed.pdf")
    other = parse_pdf(build_pdf(["Different content entirely."]), "a.pdf")

    assert first.document_id == second.document_id
    assert first.document_id.startswith("doc_") and len(first.document_id) == 16
    assert other.document_id != first.document_id


def test_scanned_pdf_without_text_is_rejected_with_ocr_note():
    with pytest.raises(NoExtractableTextError, match="OCR is not supported"):
        parse_pdf(build_pdf([None, None]), "scan.pdf")


@pytest.mark.parametrize(
    ("data", "filename", "error"),
    [
        (b"%PDF-1.4 this is not really a pdf", "broken.pdf", MalformedPDFError),
        (b"", "empty.pdf", MalformedPDFError),
        (b"hello world", "notes.pdf", UnsupportedFileTypeError),
        (b"%PDF-1.4", "image.png", UnsupportedFileTypeError),
    ],
)
def test_invalid_files_raise_clear_errors(data, filename, error):
    with pytest.raises(error):
        parse_pdf(data, filename)


def test_truncated_pdf_is_malformed(sample_pdf):
    with pytest.raises(MalformedPDFError, match="could not be parsed"):
        parse_pdf(sample_pdf[: len(sample_pdf) // 3], "truncated.pdf")


def test_file_size_limit(sample_pdf):
    with pytest.raises(FileTooLargeError, match="limit"):
        parse_pdf(sample_pdf, "big.pdf", max_bytes=100)


def test_password_protected_pdf_is_rejected(sample_pdf):
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(sample_pdf)))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(MalformedPDFError, match="password-protected"):
        parse_pdf(buffer.getvalue(), "locked.pdf")


def test_load_pdf_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_chunks_preserve_page_metadata_and_ids(sample_pdf):
    document = parse_pdf(sample_pdf, "report.pdf")

    chunks = chunk_document(document, chunk_size=60, chunk_overlap=20)

    assert {c.page_number for c in chunks} == {1, 2, 4}  # page 3 has no text
    for chunk in chunks:
        page_text = document.pages[chunk.page_number - 1].text
        assert page_text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.chunk_id == f"{document.document_id}:p{chunk.page_number}:c{chunk.chunk_index}"
        assert chunk.filename == "report.pdf"
        assert chunk.metadata == {"title": "Microgrid test report"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunk_document(document, 60, 20) == chunks  # deterministic


def test_split_text_overlaps_and_respects_word_boundaries():
    text = " ".join(f"word{i:03d}" for i in range(200))  # 8-char words

    spans = split_text(text, chunk_size=100, chunk_overlap=30)
    pieces = [text[s:e].strip() for s, e in spans]

    assert all(len(p) <= 100 for p in pieces)
    assert all(p.split()[0].startswith("word") and len(p.split()[-1]) == 7 for p in pieces)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert 0 < prev_end - next_start <= 30  # consecutive chunks overlap
    assert pieces[-1].endswith("word199")
    covered = set()
    for s, e in spans:
        covered.update(range(s, e))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_split_text_without_overlap_and_long_words():
    assert split_text("a" * 250, chunk_size=100, chunk_overlap=0) == [
        (0, 100),
        (100, 200),
        (200, 250),
    ]
    assert split_text("", 100, 10) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_split_text_rejects_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_bundled_sample_report_matches_eval_cases():
    document = load_pdf(SAMPLE_DIR / "researchpilot_sample_report.pdf")
    cases = json.loads((SAMPLE_DIR / "sample_eval_cases.json").read_text())

    assert document.page_count == 4 and not document.warnings
    assert "SYNTHETIC SAMPLE DOCUMENT" in document.pages[0].text
    assert "3.1 million" in document.pages[2].text
    for case in cases:
        assert all(1 <= page <= document.page_count for page in case["expected_pages"])
        assert case["answerable"] == bool(case["expected_pages"])
