"""Chunking: split document text into bounded, coherent pieces.

Paragraph-level splitting with a maximum size cap; oversized paragraphs
are sub-split at sentence boundaries rather than mid-sentence, so
retrieved chunks stay semantically coherent. Smaller chunks generally
improve retrieval precision at the cost of more chunks to embed and
search -- see the README for the tradeoff discussion and the
evaluation results in the accompanying paper.
"""

from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_text(text: str, min_len: int = 40, max_len: int = 900) -> list[str]:
    """Split ``text`` into chunks bounded by ``max_len`` characters.

    Parameters
    ----------
    text:
        The full document text.
    min_len:
        Discard paragraph fragments shorter than this (typically stray
        headers/whitespace, not useful retrieval targets).
    max_len:
        Maximum chunk size in characters. Paragraphs longer than this
        are sub-split at sentence boundaries.

    Returns
    -------
    list[str]
        Ordered list of chunks.
    """
    raw_chunks = [c.strip() for c in _PARAGRAPH_SPLIT.split(text) if c.strip()]
    raw_chunks = [c for c in raw_chunks if len(c) >= min_len]

    final_chunks: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= max_len:
            final_chunks.append(chunk)
            continue
        sentences = _SENTENCE_SPLIT.split(chunk)
        buf = ""
        for sentence in sentences:
            if len(buf) + len(sentence) <= max_len:
                buf = (buf + " " + sentence).strip()
            else:
                if buf:
                    final_chunks.append(buf)
                buf = sentence
        if buf:
            final_chunks.append(buf)

    return final_chunks
