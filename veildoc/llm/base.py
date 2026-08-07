"""Abstract interface for LLM backends.

Implement this to plug in any LLM -- commercial API (Claude, GPT,
Gemini) or local (Ollama, llama.cpp, any OpenAI-compatible local
server). The pipeline only depends on this interface. Note that
whichever backend you choose only ever receives the already-redacted,
minimally-scoped excerpt -- the LLM backend itself has no visibility
into the full source document.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Base class for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send ``prompt`` to the model and return its text response."""
        raise NotImplementedError
