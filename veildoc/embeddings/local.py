"""Local, on-device embedding backends via sentence-transformers.

No network call is made for embedding once the model weights are
downloaded and cached locally -- this is what makes the pipeline
genuinely client-side. BGE-base is the validated default (see the
accompanying paper's Section 6.3 for the comparison against MiniLM
and E5-large that motivated this choice).
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_E5_QUERY_PREFIX = "query: "


class LocalBGEEmbedding(EmbeddingBackend):
    """BGE-base-en-v1.5, run locally via sentence-transformers.

    Validated in the accompanying paper as matching full-LLM-quality
    retrieval on structurally distinct academic content, where a
    smaller model (MiniLM) failed. Requires ``pip install
    sentence-transformers``.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([_BGE_QUERY_PREFIX + query])[0]


class LocalE5Embedding(EmbeddingBackend):
    """E5-large-v2, run locally. Included for comparison; the paper's
    evaluation found this underperformed BGE-base on this task despite
    being larger, with misleadingly high confidence scores on wrong
    picks -- see Section 6.3. Provided for users who want to reproduce
    or extend that comparison, not as the recommended default.
    """

    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([_E5_QUERY_PREFIX + query])[0]


class SentenceTransformerEmbedding(EmbeddingBackend):
    """Generic wrapper for any sentence-transformers model, for users
    who want to plug in a model not specifically wrapped above.
    """

    def __init__(self, model_name: str, query_prefix: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([self._query_prefix + query])[0]
