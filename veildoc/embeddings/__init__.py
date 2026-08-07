from .base import EmbeddingBackend
from .local import LocalBGEEmbedding, LocalE5Embedding, SentenceTransformerEmbedding

__all__ = [
    "EmbeddingBackend",
    "LocalBGEEmbedding",
    "LocalE5Embedding",
    "SentenceTransformerEmbedding",
]

# API-based backends are imported lazily via veildoc.embeddings.api
# to avoid requiring openai/google-genai as hard dependencies.
