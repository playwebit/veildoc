"""Minimal-disclosure retrieval: select only the chunk(s) relevant to
a query, so the rest of the document never becomes candidate material
for transmission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings.base import EmbeddingBackend


@dataclass
class RetrievedChunk:
    index: int
    score: float
    text: str


class Retriever:
    """Embeds a document's chunks once, then retrieves top-k matches
    for arbitrary queries against that fixed embedding index.
    """

    def __init__(self, embedding_backend: EmbeddingBackend, chunks: list[str]):
        if not chunks:
            raise ValueError("chunks must be a non-empty list")
        self.embedding_backend = embedding_backend
        self.chunks = chunks
        self.embeddings: np.ndarray = embedding_backend.embed(chunks)

    def retrieve(self, query: str, k: int = 1) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to ``query``."""
        if k < 1:
            raise ValueError("k must be >= 1")
        query_embedding = self.embedding_backend.embed_query(query)
        similarities = self.embeddings @ query_embedding
        ranked_indices = np.argsort(similarities)[::-1][:k]
        return [
            RetrievedChunk(index=int(i), score=float(similarities[i]), text=self.chunks[i])
            for i in ranked_indices
        ]

    def rank_of(self, query: str, chunk_index: int) -> int:
        """Return the 1-indexed rank of ``chunk_index`` in the full
        similarity ranking for ``query`` -- useful for diagnosing how
        close a "miss" actually was (see the paper's Section 5.3 for
        why this matters more than a binary hit/miss).
        """
        query_embedding = self.embedding_backend.embed_query(query)
        similarities = self.embeddings @ query_embedding
        ranked_indices = list(np.argsort(similarities)[::-1])
        return ranked_indices.index(chunk_index) + 1

    def exposure_pct(self, retrieved: list[RetrievedChunk], full_document_text: str) -> float:
        """Percentage of the full document's character length contained
        in the retrieved chunk(s) -- the primary exposure metric used
        throughout the paper's evaluation.
        """
        exposed_chars = sum(len(c.text) for c in retrieved)
        return exposed_chars / len(full_document_text) * 100
