"""
Chat API: stream LLM responses with RAG context from the vector DB.
"""
import json
import logging

from flask import Blueprint, Response, jsonify, request

from config import get_config
from exceptions import ChatError, ValidationError, error_response
from middleware.rate_limit import limit
from schemas import ChatRequest

chat_bp = Blueprint("chat", __name__, url_prefix="/api/agent")


def _get_chat_service():
    from app import get_chat_service
    return get_chat_service()


def _get_llm_client():
    from app import get_llm_client
    return get_llm_client()


@chat_bp.post("/chat")
@limit(get_config().rate_limit_chat)
def api_agent_chat():
    raw = request.get_json(silent=True) or {}
    try:
        req = ChatRequest(
            message=raw.get("message"),
            googleId=raw.get("googleId") or raw.get("google_id"),
            driveUrl=raw.get("driveUrl") or raw.get("drive_url"),
            history=raw.get("history") or [],
        )
    except Exception as e:
        if hasattr(e, "errors") and callable(e.errors):
            details = [
                {"loc": list(err.get("loc", [])), "msg": str(err.get("msg", ""))}
                for err in e.errors()
            ]
            return jsonify({"error": "Validation failed", "details": details}), 400
        return jsonify({"error": str(e)}), 400

    try:
        system_prompt, chat_messages = _get_chat_service().get_system_prompt_and_messages(
            message=req.message,
            google_id=req.google_id,
            drive_url=req.drive_url,
            history=[{"role": m.role, "content": m.content} for m in req.history],
        )
    except (ValidationError, ChatError) as e:
        return jsonify(error_response(e)[0]), error_response(e)[1]

    llm = _get_llm_client()

    def generate():
        try:
            for token in llm.stream_chat(system_prompt, chat_messages):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_msg = str(e)
            logging.exception("Chat stream error: %s", err_msg)
            yield f"data: {json.dumps({'error': err_msg})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
