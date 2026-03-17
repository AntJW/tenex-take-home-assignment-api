"""
Drive load business logic: fetch from Drive, chunk, embed, and upsert into vector DB.
"""
import logging
import uuid
from typing import Any, Protocol

from config import Config, get_config
from exceptions import DriveLoadError, ValidationError
from utils import (
    chunk_text,
    fetch_drive_file,
    fetch_drive_files,
    parse_drive_url,
)


class EmbeddingsClientProtocol(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class VectorDBClientProtocol(Protocol):
    def ensure_collection(self) -> None: ...
    def delete_by_drive_url(self, google_id: str, drive_url: str) -> None: ...
    def delete_by_google_id_and_file_id(self, google_id: str, file_id: str) -> None: ...
    def upsert(self, points: list[dict[str, Any]]) -> None: ...


class DriveService:
    """Loads Google Drive folder or file into the vector DB. Dependencies are injectable for tests."""

    def __init__(
        self,
        embeddings_client: EmbeddingsClientProtocol,
        vector_db_client: VectorDBClientProtocol,
        config: Config | None = None,
    ) -> None:
        self._embeddings = embeddings_client
        self._vectordb = vector_db_client
        self._config = config or get_config()

    def load(
        self,
        drive_url: str,
        access_token: str,
        google_id: str,
    ) -> dict[str, Any]:
        """
        Parse drive URL, fetch files, chunk, embed, and upsert. Deduplicates by (google_id, file_id).
        Returns dict with folderId, files, chunksIndexed.
        """
        parsed = parse_drive_url(drive_url)
        if not parsed:
            raise ValidationError("Invalid Google Drive folder or file URL")

        url_type, resource_id = parsed

        try:
            if url_type == "folder":
                files = fetch_drive_files(resource_id, access_token)
            else:
                files = fetch_drive_file(resource_id, access_token)
        except Exception as e:
            logging.exception("Drive API error: %s", e)
            raise DriveLoadError(str(e)) from e

        if len(files) > self._config.drive_max_files_per_folder:
            raise DriveLoadError(
                f"Too many files ({len(files)}). Maximum is {self._config.drive_max_files_per_folder}.",
                user_message="Too many files in this folder. Try a smaller folder.",
            )

        self._vectordb.ensure_collection()
        self._vectordb.delete_by_drive_url(google_id, drive_url)
        for file in files:
            if file.get("id"):
                self._vectordb.delete_by_google_id_and_file_id(google_id, file["id"])

        all_chunks: list[dict[str, Any]] = []
        for file in files:
            content = (file.get("content") or "").strip()
            if not content:
                continue
            for text in chunk_text(
                content,
                chunk_size=self._config.drive_chunk_size,
                overlap=self._config.drive_chunk_overlap,
            ):
                all_chunks.append({
                    "text": text,
                    "fileName": file.get("name", "unknown"),
                    "fileId": file.get("id"),
                })

        if not all_chunks:
            return {
                "folderId": resource_id,
                "files": [{"name": f.get("name", ""), "mimeType": f.get("mimeType", "")} for f in files],
                "chunksIndexed": 0,
            }
        try:
            vectors = self._embeddings.embed_batch([c["text"] for c in all_chunks])
        except Exception as e:
            logging.exception("Embedding error: %s", e)
            raise DriveLoadError(str(e), user_message="Failed to process documents.") from e

        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vectors[i],
                "payload": {
                    "googleId": google_id,
                    "fileName": all_chunks[i]["fileName"],
                    "fileId": all_chunks[i].get("fileId"),
                    "driveUrl": drive_url,
                    "content": all_chunks[i]["text"],
                },
            }
            for i in range(len(all_chunks))
        ]
        batch_size = self._config.drive_load_batch_size
        try:
            for i in range(0, len(points), batch_size):
                self._vectordb.upsert(points[i : i + batch_size])
        except Exception as e:
            logging.exception("Vector DB upsert error: %s", e)
            raise DriveLoadError(str(e), user_message="Failed to save documents.") from e

        logging.info(
            "Indexed %s chunks from %s files for user %s",
            len(all_chunks),
            len(files),
            google_id,
        )
        return {
            "folderId": resource_id,
            "files": [{"name": f.get("name", ""), "mimeType": f.get("mimeType", "")} for f in files],
            "chunksIndexed": len(all_chunks),
        }
