"""
Embeddings client for vectorizing text. Handles empty input and response shape.
"""
import logging
import os
from typing import Any

import ollama


class EmbeddingsAPIClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self._url = host or os.getenv("EMBEDDINGS_API_URL")
        self._model = model or os.getenv("EMBEDDINGS_MODEL")
        if not self._model:
            raise ValueError("EMBEDDINGS_MODEL or model argument is required")
        self._client = ollama.Client(host=self._url)

    def embed(self, text: str) -> list[float]:
        if not (text and text.strip()):
            raise ValueError("embed() requires non-empty text")
        embed_response = self._client.embed(model=self._model, input=text.strip())
        return self._parse_embeddings_response(embed_response, single=True)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip() if t else "" for t in texts]
        if all(not t for t in cleaned):
            raise ValueError("embed_batch() requires at least one non-empty text")
        embed_response = self._client.embed(model=self._model, input=cleaned)
        return self._parse_embeddings_response(embed_response, single=False)

    def _parse_embeddings_response(self, data: Any, single: bool) -> list[float] | list[list[float]]:
        embeddings = (data or {}).get("embeddings")
        if not embeddings:
            logging.warning("Embeddings API returned no embeddings")
            raise ValueError("Embeddings API returned no embeddings")
        if single:
            return embeddings[0] if isinstance(embeddings[0], list) else list(embeddings[0])
        return [e if isinstance(e, list) else list(e) for e in embeddings]
