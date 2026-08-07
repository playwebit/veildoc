from __future__ import annotations

from .base import LLMBackend


class OpenAIBackend(LLMBackend):
    """GPT via the OpenAI API. Requires ``pip install openai``."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        """
        Parameters
        ----------
        model:
            Required, not defaulted -- OpenAI's current model lineup
            changes frequently; check https://platform.openai.com/docs/models
            for the current recommended model string rather than relying
            on a hardcoded default here.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
