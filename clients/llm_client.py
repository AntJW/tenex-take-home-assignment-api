"""
LLM client for chat and structured outputs. Validates config at init.
"""
import json
import os
from collections.abc import Generator
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.getenv("LLM_API_KEY")
        if not self._api_key:
            raise ValueError("LLM_API_KEY is not configured")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL

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
            yield from stream.text_stream

    def chat_structured(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Non-streaming chat with guaranteed JSON structure.

        Uses Anthropic structured outputs (output_config) so the response is valid JSON
        matching the given schema. Requires a supported model (e.g. Claude Sonnet 4.5+).
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": json_schema,
                },
            },
        )
        block = response.content[0]
        text = block.text if hasattr(block, "text") else ""
        return json.loads(text)
