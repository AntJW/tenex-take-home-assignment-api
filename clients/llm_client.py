import os
from typing import Generator

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._model = model or os.getenv("LLM_MODEL")

    def stream_chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Stream LLM response tokens. messages: list of { role: 'user'|'assistant', content: str }."""
        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
