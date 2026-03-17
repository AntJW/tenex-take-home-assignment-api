"""
Application-specific exceptions and consistent error response handling.
"""
from typing import Any


class AppError(Exception):
    """Base for API errors with optional status code and user-facing message."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        user_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.user_message = user_message or message
        self.details = details or {}


class ValidationError(AppError):
    """Invalid request payload or parameters (400)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            status_code=400,
            user_message=message,
            details=details,
        )


class DriveLoadError(AppError):
    """Drive load failed (e.g. permissions, API error)."""

    def __init__(
        self,
        message: str,
        user_message: str = "Failed to load Google Drive folder or file. Check your permissions.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=500,
            user_message=user_message,
            details=details or {},
        )


class ChatError(AppError):
    """Chat or retrieval failed."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        user_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            user_message=user_message or message,
            details=details or {},
        )


def error_response(error: AppError) -> tuple[dict[str, Any], int]:
    """Build a consistent JSON error response from an AppError."""
    body: dict[str, Any] = {"error": error.user_message}
    if error.details:
        body["details"] = error.details
    return body, error.status_code
