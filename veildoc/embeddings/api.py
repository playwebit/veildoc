"""API-based embedding backends.

IMPORTANT PRIVACY NOTE: using an API-based embedding backend sends
every chunk of your document to that API provider to be embedded --
including chunks that are never ultimately selected for retrieval.
This defeats the client-side privacy guarantee described in the
accompanying paper (Section 3), which assumes embeddings are computed
entirely on-device. Only use these backends if you have already
decided the document's full content can be shared with that provider;
otherwise use ``veildoc.embeddings.local`` instead.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend


class OpenAIEmbedding(EmbeddingBackend):
    """OpenAI embeddings API (e.g. text-embedding-3-small/large).

    See the module-level privacy note above before using this in a
    privacy-sensitive workflow.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = np.array([d.embedding for d in response.data])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class GeminiEmbedding(EmbeddingBackend):
    """Google Gemini embeddings API.

    See the module-level privacy note above before using this in a
    privacy-sensitive workflow.
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install google-genai") from exc
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        result = self._client.models.embed_content(model=self._model, contents=texts)
        vectors = np.array([e.values for e in result.embeddings])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)
