"""Unit tests for services with mocked clients."""
from unittest.mock import MagicMock

import pytest

from config import Config
from exceptions import DriveLoadError, ValidationError
from services.chat_service import ChatService
from services.drive_service import DriveService


@pytest.fixture
def config():
    return Config(
        drive_chunk_size=100,
        drive_chunk_overlap=20,
        drive_load_batch_size=5,
        drive_max_files_per_folder=10,
        chat_search_limit=3,
        chat_max_history_messages=5,
    )


class TestDriveService:
    def test_invalid_drive_url_raises(self, config):
        embeddings = MagicMock()
        vectordb = MagicMock()
        svc = DriveService(embeddings, vectordb, config)
        with pytest.raises(ValidationError):
            svc.load(
                drive_url="https://not-drive.example.com/",
                access_token="token",
                google_id="user1",
            )
        vectordb.ensure_collection.assert_not_called()

    def test_valid_url_but_fetch_fails_raises_drive_load_error(self, config):
        from unittest.mock import patch

        embeddings = MagicMock()
        embeddings.embed_batch.return_value = [[0.1] * 768]
        vectordb = MagicMock()
        with patch("services.drive_service.fetch_drive_files") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            svc = DriveService(embeddings, vectordb, config)
            with pytest.raises(DriveLoadError):
                svc.load(
                    drive_url="https://drive.google.com/drive/folders/abc123",
                    access_token="token",
                    google_id="user1",
                )


class TestChatService:
    def test_returns_system_prompt_and_messages(self, config):
        embeddings = MagicMock()
        embeddings.embed.return_value = [0.1] * 768
        vectordb = MagicMock()
        vectordb.search.return_value = [
            {
                "score": 0.9,
                "payload": {
                    "fileName": "doc.txt",
                    "fileId": "f1",
                    "content": "Some content",
                    "driveUrl": "https://drive.google.com/",
                },
            }
        ]
        svc = ChatService(embeddings, vectordb, config)
        system_prompt, messages = svc.get_system_prompt_and_messages(
            message="What is this?",
            google_id="user1",
            drive_url=None,
            history=[],
        )
        assert "doc.txt" in system_prompt
        assert "Some content" in system_prompt
        assert messages == [{"role": "user", "content": "What is this?"}]
        embeddings.embed.assert_called_once_with("What is this?")
        vectordb.search.assert_called_once()

    def test_empty_search_still_returns_prompt(self, config):
        embeddings = MagicMock()
        embeddings.embed.return_value = [0.1] * 768
        vectordb = MagicMock()
        vectordb.search.return_value = []
        svc = ChatService(embeddings, vectordb, config)
        system_prompt, messages = svc.get_system_prompt_and_messages(
            message="Hi",
            google_id="user1",
            drive_url="https://drive.google.com/folders/x",
            history=[],
        )
        assert "No relevant documents" in system_prompt
        assert len(messages) == 1
