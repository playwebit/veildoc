"""Local LLM backends -- genuinely zero-network options.

For users who want no external API call at all, not even for the
already-redacted excerpt (see the paper's Section 6 discussion of the
FHE/TEE tradeoffs that motivate this as the practical fully-local
option today).
"""

from __future__ import annotations

from .base import LLMBackend


class OllamaBackend(LLMBackend):
    """Local model served by Ollama (https://ollama.com), e.g.
    ``ollama run llama3`` running on localhost. No network call leaves
    the machine at all with this backend.
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install requests") from exc
        self._model = model
        self._host = host.rstrip("/")

    def generate(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


class OpenAICompatibleBackend(LLMBackend):
    """Any server exposing an OpenAI-compatible chat completions
    endpoint (llama.cpp server, vLLM, LM Studio, text-generation-webui,
    etc.). Point ``base_url`` at your local server.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install openai") from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
