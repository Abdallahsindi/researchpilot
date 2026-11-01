"""PDF validation and page-level text extraction.

Only PDFs with an embedded text layer are supported. Scanned pages are images; extracting
text from them needs OCR, which this project does not implement. Such pages are reported in
`warnings` rather than silently dropped.
"""

import hashlib
import io
import logging
import re
from pathlib import Path

from app.rag.models import PageText, ParsedDocument

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
OCR_NOTE = "OCR is not supported, so scanned or image-only pages cannot be read."


class DocumentError(Exception):
    """Base class for documents that cannot be ingested."""


class UnsupportedFileTypeError(DocumentError):
    pass


class FileTooLargeError(DocumentError):
    pass


class MalformedPDFError(DocumentError):
    pass


class NoExtractableTextError(DocumentError):
    pass


def document_id_for(data: bytes) -> tuple[str, str]:
    sha256 = hashlib.sha256(data).hexdigest()
    return f"doc_{sha256[:12]}", sha256


def _clean(text: str) -> str:
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _format_pages(numbers: list[int]) -> str:
    return ", ".join(str(n) for n in numbers[:20]) + (" ..." if len(numbers) > 20 else "")


def parse_pdf(data: bytes, filename: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> ParsedDocument:
    """Validate a PDF and extract its text page by page.

    Raises a `DocumentError` subclass with a user-facing message when the file is not a PDF,
    is too large, cannot be parsed, or contains no extractable text at all.
    """
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(f"'{filename}' is not a .pdf file.")
    if not data:
        raise MalformedPDFError(f"'{filename}' is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB."
        )
    # The PDF spec allows the header anywhere in the first 1024 bytes.
    if b"%PDF-" not in data[:1024]:
        raise UnsupportedFileTypeError(f"'{filename}' does not have a PDF header.")

    from pypdf import PdfReader  # imported here so importing this module stays cheap

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise MalformedPDFError(f"'{filename}' is password-protected.")
        raw_pages = list(reader.pages)
        raw_metadata = reader.metadata or {}
    except DocumentError:
        raise
    except Exception as exc:  # pypdf raises a variety of errors for corrupt files
        raise MalformedPDFError(f"'{filename}' could not be parsed as a PDF: {exc}") from exc

    if not raw_pages:
        raise NoExtractableTextError(f"'{filename}' contains no pages.")

    warnings: list[str] = []
    pages: list[PageText] = []
    for number, page in enumerate(raw_pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception as exc:
            logger.warning("Text extraction failed on %s page %d: %s", filename, number, exc)
            warnings.append(f"Page {number}: text extraction failed ({exc}).")
            text = ""
        pages.append(PageText(page_number=number, text=text))

    empty = [p.page_number for p in pages if not p.has_text]
    if len(empty) == len(pages):
        raise NoExtractableTextError(
            f"No extractable text found in any of the {len(pages)} page(s) of '{filename}'. "
            f"The PDF is probably scanned or image-only. {OCR_NOTE}"
        )
    if empty:
        warnings.append(f"Page(s) {_format_pages(empty)} have no extractable text. {OCR_NOTE}")

    document_id, sha256 = document_id_for(data)
    metadata = {
        key.lstrip("/").lower(): str(value)
        for key, value in raw_metadata.items()
        if key in ("/Title", "/Author", "/Subject") and value
    }
    return ParsedDocument(
        document_id=document_id,
        filename=filename,
        sha256=sha256,
        size_bytes=len(data),
        page_count=len(pages),
        pages=pages,
        metadata=metadata,
        warnings=warnings,
    )


def load_pdf(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> ParsedDocument:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    return parse_pdf(path.read_bytes(), path.name, max_bytes=max_bytes)
