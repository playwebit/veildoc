"""Abstract interface for embedding backends.

Implement this to plug in any embedding model -- local (sentence-
transformers, ONNX, etc.) or API-based (OpenAI, Gemini, Voyage, ...).
The pipeline only depends on this interface, never on a specific
model, so swapping embedding models never requires touching pipeline
or retrieval code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Base class for all embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Parameters
        ----------
        texts:
            Texts to embed (chunks, or a single-element list for a query).

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), embedding_dim)``. Implementations
            should L2-normalize embeddings so downstream cosine similarity
            reduces to a dot product.
        """
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Override if your model needs a different
        prefix/instruction format for queries vs. documents (e.g. BGE, E5).
        """
        return self.embed([query])[0]
