"""Split page text into overlapping chunks.

Chunks never span two pages, so every chunk maps to exactly one page number. Sizes are
measured in characters. The defaults (500 chars, roughly 100-125 tokens) stay well under the
256-token input limit of all-MiniLM-L6-v2, and on the sample report they gave more focused
matches than 800-char chunks.
"""

from app.rag.models import DocumentChunk, ParsedDocument

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def _last_whitespace(text: str, lo: int, hi: int) -> int:
    for i in range(hi - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) spans that cover `text`, breaking on whitespace where possible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    spans: list[tuple[int, int]] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Avoid cutting a word in half, unless the only break is very early in the chunk.
            cut = _last_whitespace(text, start + chunk_size // 2, end)
            if cut > start:
                end = cut
        spans.append((start, end))
        if end >= length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        # Begin the overlap at a word boundary.
        while 0 < next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
        while start < length and text[start].isspace():
            start += 1
    return spans


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for page in document.pages:
        if not page.has_text:
            continue
        index = 0
        for start, end in split_text(page.text, chunk_size, chunk_overlap):
            raw = page.text[start:end]
            text = raw.strip()
            if not text:
                continue
            start += len(raw) - len(raw.lstrip())
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}:p{page.page_number}:c{index}",
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=text,
                    start_char=start,
                    end_char=start + len(text),
                    metadata=document.metadata,
                )
            )
            index += 1
    return chunks
