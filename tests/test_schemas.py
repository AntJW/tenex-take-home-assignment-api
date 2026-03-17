"""Tests for request schemas."""
import pytest
from pydantic import ValidationError

from schemas import ChatRequest, DriveLoadRequest


class TestDriveLoadRequest:
    def test_valid(self):
        req = DriveLoadRequest(
            driveUrl="https://drive.google.com/drive/folders/abc",
            accessToken="token",
            googleId="user1",
        )
        assert req.drive_url == "https://drive.google.com/drive/folders/abc"
        assert req.google_id == "user1"

    def test_missing_drive_url_raises(self):
        with pytest.raises(ValidationError):
            DriveLoadRequest(accessToken="t", googleId="u")

    def test_missing_token_raises(self):
        with pytest.raises(ValidationError):
            DriveLoadRequest(driveUrl="https://x", googleId="u")

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            DriveLoadRequest(driveUrl="https://x", accessToken="t", googleId="")


class TestChatRequest:
    def test_valid(self):
        req = ChatRequest(message="Hi", googleId="user1")
        assert req.message == "Hi"
        assert req.google_id == "user1"
        assert req.drive_url is None
        assert req.history == []

    def test_drive_url_optional(self):
        req = ChatRequest(message="Hi", googleId="u", driveUrl="https://drive.google.com/...")
        assert req.drive_url == "https://drive.google.com/..."

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(googleId="u")

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="", googleId="u")
