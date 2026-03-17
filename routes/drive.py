"""
Drive load API: load Google Drive folder or file into the vector DB.
"""
import logging

from flask import Blueprint, jsonify, request

from config import get_config
from exceptions import DriveLoadError, ValidationError, error_response
from middleware.rate_limit import limit
from schemas import DriveLoadRequest

drive_bp = Blueprint("drive", __name__, url_prefix="/api/drive")


def _get_drive_service():
    """Lazy import to avoid circular imports and allow app to inject deps."""
    from app import get_drive_service
    return get_drive_service()


@drive_bp.post("/load")
@limit(get_config().rate_limit_drive_load)
def api_drive_load():
    raw = request.get_json(silent=True) or {}
    try:
        req = DriveLoadRequest(
            driveUrl=raw.get("driveUrl") or raw.get("drive_url"),
            accessToken=raw.get("accessToken") or raw.get("access_token"),
            googleId=raw.get("googleId") or raw.get("google_id"),
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
        result = _get_drive_service().load(
            drive_url=req.drive_url,
            access_token=req.access_token,
            google_id=req.google_id,
        )
        return jsonify(result)
    except ValidationError as e:
        return jsonify(error_response(e)[0]), error_response(e)[1]
    except DriveLoadError as e:
        logging.warning("Drive load error: %s", e)
        return jsonify(error_response(e)[0]), error_response(e)[1]
