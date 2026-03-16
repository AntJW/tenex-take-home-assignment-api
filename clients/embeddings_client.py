import os
import ollama


class EmbeddingsAPIClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self._url = host or os.getenv("EMBEDDINGS_API_URL")
        self._model = model or os.getenv("EMBEDDINGS_MODEL")
        self._client = ollama.Client(host=self._url)

    def embed(self, text: str) -> list[float]:
        embed_response = self._client.embed(model=self._model, input=text)
        return embed_response["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embed_response = self._client.embed(model=self._model, input=texts)
        return embed_response["embeddings"]
