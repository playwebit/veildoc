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


def _hard_wrap(text: str, max_len: int) -> list[str]:
    """Split text that has no usable sentence boundary into word-boundary
    pieces no longer than max_len. Last-resort fallback for run-on
    sentences or text containing abbreviations that fool sentence
    detection (e.g. "et al.", "Fig. 3") into treating a long span as
    a single "sentence".
    """
    words = text.split(" ")
    pieces: list[str] = []
    buf = ""
    for word in words:
        candidate = (buf + " " + word).strip() if buf else word
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = word
    if buf:
        pieces.append(buf)
    return pieces


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
        Maximum chunk size in characters, enforced as a hard cap.
        Paragraphs longer than this are first sub-split at sentence
        boundaries; any individual "sentence" that is itself still
        longer than max_len (e.g. a run-on sentence, or text containing
        abbreviations that fool sentence-boundary detection) is further
        split at word boundaries so no returned chunk ever exceeds
        max_len.

    Returns
    -------
    list[str]
        Ordered list of chunks, each guaranteed to be <= max_len characters.
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
            candidate = (buf + " " + sentence).strip() if buf else sentence
            if len(candidate) <= max_len:
                buf = candidate
            else:
                if buf:
                    final_chunks.append(buf)
                if len(sentence) > max_len:
                    # single sentence too long on its own -- hard-wrap it
                    final_chunks.extend(_hard_wrap(sentence, max_len))
                    buf = ""
                else:
                    buf = sentence
        if buf:
            if len(buf) > max_len:
                final_chunks.extend(_hard_wrap(buf, max_len))
            else:
                final_chunks.append(buf)

    return final_chunks
