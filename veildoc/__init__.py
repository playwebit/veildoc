"""
veildoc: a client-side, minimal-disclosure privacy pipeline for using
commercial or local LLMs on sensitive source documents.

Combines minimal-disclosure retrieval (only the relevant excerpt of a
document is ever transmitted) with entity redaction (identifying
information is stripped before transmission and restored locally in
the response).

Quick start:

    from veildoc import Pipeline
    from veildoc.embeddings import LocalBGEEmbedding
    from veildoc.llm import AnthropicBackend

    pipeline = Pipeline(
        embedding_backend=LocalBGEEmbedding(),
        llm_backend=AnthropicBackend(api_key="..."),
    )
    pipeline.load_document("paper.pdf")
    answer = pipeline.ask("Can you help me improve the clarity of the methodology?")
    print(answer)
"""

from .pipeline import Pipeline
from .redaction import Redactor
from .chunking import chunk_text
from .parsing import extract_pdf_text

__version__ = "0.1.0"
__all__ = ["Pipeline", "Redactor", "chunk_text", "extract_pdf_text"]
