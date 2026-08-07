from __future__ import annotations

from .base import LLMBackend


class AnthropicBackend(LLMBackend):
    """Claude via the Anthropic API. Requires ``pip install anthropic``."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5", max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))
