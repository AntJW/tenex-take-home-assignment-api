from qdrant_client import models, QdrantClient
from qdrant_client.conversions.common_types import Filter
from clients.embeddings_client import EmbeddingsAPIClient
import os
from pydantic import BaseModel
from typing import Literal, Optional, Any
import uuid

DEFAULT_VECTOR_DB_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "drive_documents"
VECTOR_SIZE = 768  # nomic-embed-text:v1.5


class Document(BaseModel):
    # content is the data to be vectorized
    content: str
    type: Literal["conversation_transcript"]
    # userId is used for multitenancy
    userId: str
    customerId: Optional[str] = None


class VectorDBClient:
    def __init__(self, base_url: str | None = None, collection: str | None = None):
        self._url = (base_url or os.getenv("VECTOR_DB_URL") or DEFAULT_VECTOR_DB_URL).rstrip("/")
        self._collection_name = collection or os.getenv("VECTOR_DB_COLLECTION") or DEFAULT_COLLECTION

        # TODO: Uncomment this before deploying to production
        # self._api_key = os.getenv("VECTOR_DB_API_KEY")
        # self._client = QdrantClient(url=self._url, api_key=self._api_key)

        # TODO: Delete this line before deploying to production
        self._client = QdrantClient(url=self._url)

    def upload_documents(self, documents: list[dict]):
        self._client.upload_points(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()), vector=EmbeddingsAPIClient().embed(doc["content"]), payload=doc
                )
                for idx, doc in enumerate(documents)
            ],
        )

    def create_collection(self, distance: models.Distance = models.Distance.COSINE):
        self._client.create_collection(
            collection_name=self._collection_name, vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=distance))

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist (for drive documents)."""
        try:
            self._client.get_collection(self._collection_name)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )

    def upsert(self, points: list[dict[str, Any]]) -> None:
        """Upsert points with payloads { googleId, fileName, driveUrl, content }."""
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def search(
        self, vector: list[float], google_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search by vector filtered by googleId. Returns list of { id, score, payload }."""
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="googleId", match=models.MatchValue(value=google_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": p.id, "score": p.score or 0.0, "payload": p.payload or {}}
            for p in results.points
        ]

    def delete_by_drive_url(self, google_id: str, drive_url: str) -> None:
        """Delete points for this user and Drive folder URL."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="googleId", match=models.MatchValue(value=google_id)),
                        models.FieldCondition(key="driveUrl", match=models.MatchValue(value=drive_url)),
                    ]
                )
            ),
        )

    def query(self, query: str, limit: int = 10, query_filter: Filter = None):
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=EmbeddingsAPIClient().embed(query),
            limit=limit,
            query_filter=query_filter
        )
        return response.points  # Return the points list, not the response object
