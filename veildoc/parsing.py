"""Document parsing: extract plain text from source documents.

Currently supports PDF via PyMuPDF. Additional formats (docx, plain
text, html) can be added following the same function signature:
``(path: str) -> str``.
"""

from __future__ import annotations

import os


def extract_pdf_text(path: str) -> str:
    """Extract plain text from a PDF file, page by page.

    Requires PyMuPDF (``pip install pymupdf``).

    Parameters
    ----------
    path:
        Path to a local PDF file.

    Returns
    -------
    str
        The full extracted text, with pages joined by double newlines.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install with: pip install pymupdf"
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    doc = fitz.open(path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages)


def extract_text(path: str) -> str:
    """Extract text from a document, dispatching on file extension.

    Currently only ``.pdf`` is supported. Raises ``ValueError`` for
    unsupported extensions, so callers get a clear error rather than
    a confusing downstream failure.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_text(path)
    raise ValueError(
        f"Unsupported file extension: {ext!r}. Currently supported: .pdf"
    )
