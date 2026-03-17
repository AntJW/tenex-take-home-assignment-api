"""
Rate limiting. Uses Flask-Limiter when available; no-op otherwise.
"""
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter: Limiter | None = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
    )
except ImportError:
    limiter = None


def limit(limit_string: str) -> Callable[[F], F]:
    """Decorator to apply rate limit. No-op if Flask-Limiter is not installed."""

    def decorator(view: F) -> F:
        if limiter is not None:
            return limiter.limit(limit_string)(view)  # type: ignore
        return view

    return decorator
