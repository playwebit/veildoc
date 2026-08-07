from __future__ import annotations

from .base import LLMBackend


class GeminiBackend(LLMBackend):
    """Gemini via the Google GenAI API. Requires ``pip install google-genai``.

    Note: the free tier's daily request quota varies significantly by
    model -- the "flash-lite" variants typically allow substantially
    more requests/day than the full "flash" models. Check your quota
    at https://aistudio.google.com if you hit rate limits.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install with: pip install google-genai") from exc
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text
