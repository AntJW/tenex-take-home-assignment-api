"""
Pytest fixtures: app with injected mocks for easy testing.
"""
from typing import Any
from unittest.mock import MagicMock

import pytest

from app import create_app
from config import Config


@pytest.fixture
def app_config() -> Config:
    """Config with test-friendly limits."""
    return Config(
        drive_load_batch_size=10,
        drive_chunk_size=500,
        drive_chunk_overlap=50,
        drive_max_files_per_folder=20,
        chat_search_limit=5,
        chat_max_history_messages=10,
        rate_limit_drive_load="100 per minute",
        rate_limit_chat="100 per minute",
    )


@pytest.fixture
def mock_embeddings():
    """Mock embeddings client: returns fixed-size vectors."""
    m = MagicMock()
    m.embed.return_value = [0.1] * 768
    m.embed_batch.return_value = [[0.1] * 768]
    return m


@pytest.fixture
def mock_vectordb():
    """Mock vector DB: no-op ensure_collection, delete; search returns empty or controllable hits."""
    m = MagicMock()
    m.search.return_value = []
    return m


@pytest.fixture
def mock_llm():
    """Mock LLM client: stream yields a single token."""
    m = MagicMock()
    m.stream_chat.return_value = iter(["Hello"])
    return m


@pytest.fixture
def app(
    app_config: Config,
    mock_embeddings: MagicMock,
    mock_vectordb: MagicMock,
    mock_llm: MagicMock,
) -> Any:
    """Flask app with all external services mocked."""
    return create_app(
        config=app_config,
        drive_service_factory=lambda: _drive_service(mock_embeddings, mock_vectordb, app_config),
        chat_service_factory=lambda: _chat_service(mock_embeddings, mock_vectordb, app_config),
        llm_client_factory=lambda: mock_llm,
    )


def _drive_service(embeddings, vectordb, config):
    from services.drive_service import DriveService
    return DriveService(embeddings, vectordb, config)


def _chat_service(embeddings, vectordb, config):
    from services.chat_service import ChatService
    return ChatService(embeddings, vectordb, config)


@pytest.fixture
def client(app: Any):
    """Flask test client."""
    return app.test_client()
