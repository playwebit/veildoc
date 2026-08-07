"""High-level pipeline: the main entry point most users want.

Combines parsing, chunking, embedding, retrieval, and redaction into
a single object, with an optional LLM backend for end-to-end query
answering. If no LLM backend is provided, ``ask()`` returns the
redacted excerpt for you to paste manually into any chat interface
instead -- the pipeline never requires a specific AI provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import chunk_text
from .embeddings.base import EmbeddingBackend
from .llm.base import LLMBackend
from .parsing import extract_text
from .redaction import Redactor, restore
from .retrieval import Retriever


@dataclass
class QueryResult:
    query: str
    redacted_excerpt: str
    token_map: dict[str, str]
    exposure_pct: float
    entities_redacted: int
    retrieved_chunk_indices: list[int]
    ai_response: str | None = None
    restored_response: str | None = None


class Pipeline:
    """The main pipeline object.

    Example
    -------
    >>> from veildoc import Pipeline
    >>> from veildoc.embeddings import LocalBGEEmbedding
    >>> pipeline = Pipeline(embedding_backend=LocalBGEEmbedding())
    >>> pipeline.load_document("paper.pdf")
    >>> result = pipeline.ask("Can you help me improve the clarity of the methodology?")
    >>> print(result.redacted_excerpt)  # paste this into any chat AI yourself
    """

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        llm_backend: LLMBackend | None = None,
        redactor: Redactor | None = None,
        chunk_max_len: int = 900,
        k: int = 1,
    ):
        """
        Parameters
        ----------
        embedding_backend:
            Required. Any ``EmbeddingBackend`` implementation -- see
            ``veildoc.embeddings`` for local (recommended) and API
            options.
        llm_backend:
            Optional. If provided, ``ask()`` sends the redacted excerpt
            to this backend and restores the response automatically.
            If omitted, ``ask()`` returns just the redacted excerpt for
            you to use however you like -- no AI provider is required
            to use this pipeline.
        redactor:
            Optional. Defaults to ``Redactor()`` with default settings.
            Pass a custom-configured ``Redactor`` to change entity
            types, allowlist, or confidence threshold.
        chunk_max_len:
            Maximum chunk size in characters (see ``chunking.chunk_text``).
        k:
            Number of chunks to retrieve per query.
        """
        self.embedding_backend = embedding_backend
        self.llm_backend = llm_backend
        self.redactor = redactor or Redactor()
        self.chunk_max_len = chunk_max_len
        self.k = k

        self.document_text: str | None = None
        self.chunks: list[str] = []
        self.retriever: Retriever | None = None

    def load_document(self, path: str) -> None:
        """Parse, chunk, and embed a document. Must be called before
        ``ask()``. All of this runs locally -- nothing is transmitted
        at this stage.
        """
        self.document_text = extract_text(path)
        self.chunks = chunk_text(self.document_text, max_len=self.chunk_max_len)
        if not self.chunks:
            raise ValueError(
                f"No chunks extracted from {path!r} -- the document may be "
                "empty, image-only (needs OCR), or in an unsupported format."
            )
        self.retriever = Retriever(self.embedding_backend, self.chunks)

    def load_text(self, text: str) -> None:
        """Like ``load_document``, but for text you already have in
        memory rather than a file on disk.
        """
        self.document_text = text
        self.chunks = chunk_text(self.document_text, max_len=self.chunk_max_len)
        if not self.chunks:
            raise ValueError("No chunks extracted from the provided text.")
        self.retriever = Retriever(self.embedding_backend, self.chunks)

    def ask(self, query: str, send_to_llm: bool | None = None) -> QueryResult:
        """Retrieve the relevant excerpt, redact it, and optionally
        send it to the configured LLM backend.

        Parameters
        ----------
        query:
            The question to ask about the loaded document.
        send_to_llm:
            If ``True``, sends the redacted excerpt to ``self.llm_backend``
            (raises if none was configured). If ``False``, only returns
            the redacted excerpt without sending anywhere. Defaults to
            ``True`` if an ``llm_backend`` was configured, ``False``
            otherwise.
        """
        if self.retriever is None:
            raise RuntimeError("Call load_document() or load_text() before ask().")

        if send_to_llm is None:
            send_to_llm = self.llm_backend is not None
        if send_to_llm and self.llm_backend is None:
            raise RuntimeError(
                "send_to_llm=True but no llm_backend was configured. "
                "Pass one to Pipeline(...), or call ask(query, send_to_llm=False)."
            )

        retrieved = self.retriever.retrieve(query, k=self.k)

        combined_redacted = []
        combined_token_map: dict[str, str] = {}
        for chunk in retrieved:
            result = self.redactor.redact(chunk.text)
            combined_redacted.append(result.redacted_text)
            combined_token_map.update(result.token_map)

        redacted_excerpt = "\n\n---\n\n".join(combined_redacted)
        exposure_pct = self.retriever.exposure_pct(retrieved, self.document_text)

        ai_response = None
        restored_response = None
        if send_to_llm:
            prompt = f"Question: {query}\n\nRelevant excerpt:\n{redacted_excerpt}"
            ai_response = self.llm_backend.generate(prompt)
            restored_response = restore(ai_response, combined_token_map)

        return QueryResult(
            query=query,
            redacted_excerpt=redacted_excerpt,
            token_map=combined_token_map,
            exposure_pct=exposure_pct,
            entities_redacted=len(combined_token_map),
            retrieved_chunk_indices=[c.index for c in retrieved],
            ai_response=ai_response,
            restored_response=restored_response,
        )
