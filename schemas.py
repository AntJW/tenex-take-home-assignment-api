"""
Request/response validation schemas for the API.
"""
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DriveLoadRequest(BaseModel):
    """Payload for POST /api/drive/load."""

    drive_url: str = Field(..., alias="driveUrl", min_length=1)
    access_token: str = Field(..., alias="accessToken", min_length=1)
    google_id: str = Field(..., alias="googleId", min_length=1)

    @field_validator("drive_url", "access_token", "google_id", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        if v is None:
            raise ValueError("field is required")
        return str(v).strip()

    class Config:
        populate_by_name = True


class ChatMessage(BaseModel):
    """Single message in chat history."""

    role: str = "user"
    content: str = ""

    @field_validator("role")
    @classmethod
    def role_allowed(cls, v: str) -> str:
        v = (v or "user").strip().lower()
        if v not in ("user", "assistant"):
            return "user"
        return v


class ChatRequest(BaseModel):
    """Payload for POST /api/agent/chat."""

    message: str = Field(..., min_length=1)
    google_id: str = Field(..., alias="googleId", min_length=1)
    drive_url: str | None = Field(None, alias="driveUrl")
    history: list[ChatMessage] = Field(default_factory=list, max_length=100)

    @field_validator("message", "google_id", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        if v is None:
            raise ValueError("field is required")
        return str(v).strip()

    @field_validator("drive_url", mode="before")
    @classmethod
    def coerce_drive_url(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        return str(v).strip()

    @field_validator("message")
    @classmethod
    def message_not_too_long(cls, v: str, info: Any) -> str:
        max_len = 32_000
        if len(v) > max_len:
            raise ValueError(f"message must be at most {max_len} characters")
        return v

    class Config:
        populate_by_name = True
