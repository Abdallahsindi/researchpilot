"""Minimal PDF writer for tests and the sample document. No third-party dependencies.

Produces real PDF files with a text layer (Helvetica, Latin-1 text), so pypdf extracts
exactly what was written. `None` as page text creates a page with no text layer, which
looks the same to the extractor as a scanned image page. Lines indented with four spaces
are drawn unwrapped in Courier, which keeps simple text tables aligned.
"""

import textwrap


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(text: str, title: str | None) -> bytes:
    ops = ["BT", "/F1 11 Tf", "15 TL", "72 740 Td"]
    if title:
        ops += ["/F2 15 Tf", f"({_escape(title)}) Tj", "T*", "T*", "/F1 11 Tf"]
    for paragraph in text.split("\n"):
        if paragraph.startswith("    "):  # table row: monospace, no wrapping
            ops += ["/F3 9 Tf", f"({_escape(paragraph[4:])}) Tj", "T*", "/F1 11 Tf"]
            continue
        lines = textwrap.wrap(paragraph, width=88) or [""]
        for line in lines:
            ops += [f"({_escape(line)}) Tj", "T*"]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_pdf(
    pages: list[str | None],
    *,
    page_titles: list[str | None] | None = None,
    metadata: dict[str, str] | None = None,
) -> bytes:
    titles = page_titles or [None] * len(pages)
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in once the page tree exists
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    # Only PDFs with table rows get the Courier font, so older generated files stay byte-identical.
    has_table = any(t and any(line.startswith("    ") for line in t.split("\n")) for t in pages)
    mono = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>") if has_table else None
    fonts = f"/F1 {font} 0 R /F2 {bold} 0 R" + (f" /F3 {mono} 0 R" if mono else "")
    page_ids = []
    for text, title in zip(pages, titles, strict=True):
        stream = _content_stream(text, title) if text is not None else b""
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                (
                    f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << {fonts} >> >> "
                    f"/Contents {content} 0 R >>"
                ).encode()
            )
        )
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    info = None
    if metadata:
        entries = " ".join(f"/{k} ({_escape(v)})" for k, v in metadata.items())
        info = add(f"<< {entries} >>".encode("latin-1"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog} 0 R"
    trailer += f" /Info {info} 0 R >>" if info else " >>"
    out += f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
