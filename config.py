"""
Application configuration loaded from environment with sensible defaults.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable config for the application."""

    # Drive / indexing
    drive_load_batch_size: int = 100
    drive_chunk_size: int = 1000
    drive_chunk_overlap: int = 200
    drive_max_files_per_folder: int = 500

    # Chat
    chat_search_limit: int = 10
    chat_max_history_messages: int = 50
    chat_max_message_length: int = 32_000

    # Rate limiting (per IP)
    rate_limit_drive_load: str = "20 per minute"
    rate_limit_chat: str = "60 per minute"


def get_config() -> Config:
    """Return application config. Override in tests by patching or passing explicit config."""
    return Config(
        rate_limit_drive_load=os.getenv("RATE_LIMIT_DRIVE_LOAD", "20 per minute"),
        rate_limit_chat=os.getenv("RATE_LIMIT_CHAT", "60 per minute"),
    )
