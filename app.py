"""
Application factory and dependency wiring. Injectable dependencies for testing.
"""
from collections.abc import Callable
from typing import Any

from flask import Flask, jsonify
from flask_cors import CORS
from pydantic import ValidationError as PydanticValidationError

from config import Config, get_config
from exceptions import AppError, error_response
from routes import chat_bp, drive_bp

# Injected by create_app when provided; otherwise lazy-created on first use
_drive_service_override: Any | None = None
_chat_service_override: Any | None = None
_llm_client_override: Any | None = None
_drive_service_default: Any | None = None
_chat_service_default: Any | None = None
_llm_client_default: Any | None = None


def get_drive_service():
    global _drive_service_default
    if _drive_service_override is not None:
        return _drive_service_override
    if _drive_service_default is None:
        from clients.embeddings_client import EmbeddingsAPIClient
        from clients.vector_db_client import VectorDBClient
        from services.drive_service import DriveService
        _drive_service_default = DriveService(
            EmbeddingsAPIClient(),
            VectorDBClient(),
            get_config(),
        )
    return _drive_service_default


def get_chat_service():
    global _chat_service_default
    if _chat_service_override is not None:
        return _chat_service_override
    if _chat_service_default is None:
        from clients.embeddings_client import EmbeddingsAPIClient
        from clients.vector_db_client import VectorDBClient
        from services.chat_service import ChatService
        _chat_service_default = ChatService(
            EmbeddingsAPIClient(),
            VectorDBClient(),
            get_config(),
        )
    return _chat_service_default


def get_llm_client():
    global _llm_client_default
    if _llm_client_override is not None:
        return _llm_client_override
    if _llm_client_default is None:
        from clients.llm_client import LLMClient
        _llm_client_default = LLMClient()
    return _llm_client_default


def create_app(
    config: Config | None = None,
    drive_service_factory: Callable[[], Any] | None = None,
    chat_service_factory: Callable[[], Any] | None = None,
    llm_client_factory: Callable[[], Any] | None = None,
) -> Flask:
    """
    Application factory. Inject factories for testing to override default services/clients.
    """
    global _drive_service_override, _chat_service_override, _llm_client_override
    app = Flask(__name__)
    app.config["config"] = config or get_config()

    if drive_service_factory is not None:
        _drive_service_override = drive_service_factory()
    if chat_service_factory is not None:
        _chat_service_override = chat_service_factory()
    if llm_client_factory is not None:
        _llm_client_override = llm_client_factory()

    CORS(app)

    from middleware.rate_limit import limiter
    if limiter is not None:
        limiter.init_app(app)

    app.register_blueprint(drive_bp)
    app.register_blueprint(chat_bp)

    @app.errorhandler(AppError)
    def handle_app_error(e: AppError):
        body, code = error_response(e)
        return jsonify(body), code

    @app.errorhandler(PydanticValidationError)
    def handle_pydantic_validation(e: PydanticValidationError):
        return jsonify({"error": "Validation failed", "details": e.errors()}), 400

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(429)
    def rate_limited(_e):
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    @app.errorhandler(500)
    def internal_error(_e):
        return jsonify({"error": "Internal server error"}), 500

    return app
