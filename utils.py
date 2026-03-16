"""
Shared utilities for the API: Google Drive handling and text chunking.
"""
import re
import logging

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def parse_folder_id(url: str) -> str | None:
    """
    Extract the Google Drive folder ID from a Drive folder URL.

    Use case: The /api/drive/load endpoint receives a full folder URL (e.g. from the
    frontend). The Drive API requires the folder ID to list files, so we parse it
    from paths like ".../folders/abc123xyz".
    """
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def parse_drive_url(url: str) -> tuple[str, str] | None:
    """
    Parse a Drive URL into type and ID.

    Accepts:
    - Folder URLs: https://drive.google.com/drive/folders/FOLDER_ID
    - Drive file URLs: https://drive.google.com/file/d/FILE_ID/view
    - Docs/Sheets/Slides URLs: https://docs.google.com/document/d/FILE_ID/edit...

    Returns ("folder", id) or ("file", id), or None if the URL is not recognized.
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    # Drive IDs are alphanumeric with - _ and sometimes .
    id_pattern = r"[a-zA-Z0-9_.-]+"
    folder_match = re.search(r"/folders/(" + id_pattern + r")", url)
    if folder_match:
        return ("folder", folder_match.group(1))
    # drive.google.com/file/d/ID or docs.google.com/document|spreadsheets|presentation/d/ID
    file_match = re.search(r"/(?:file|document|spreadsheets|presentation)/d/(" + id_pattern + r")", url)
    if file_match:
        return ("file", file_match.group(1))
    return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks of a maximum size.

    Use case: Long documents exceed embedding model input limits and hurt retrieval
    quality. Chunking keeps each piece within bounds (e.g. 1000 chars). Overlap
    avoids cutting mid-sentence and preserves context at chunk boundaries when
    we later search and pass excerpts to the LLM.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += chunk_size - overlap
    return chunks


def fetch_drive_files(folder_id: str, access_token: str) -> list[dict]:
    """
    Fetch metadata and text content of all supported files in a Google Drive folder.

    Use case: For /api/drive/load we need to index folder contents. This lists files
    in the folder (using the Drive API with the user's OAuth access token), then
    exports or downloads each file as text: Google Docs/Sheets/Slides are exported
    to plain text or CSV; native text, JSON, and PDF are downloaded and decoded.
    Returns a list of dicts with name, content, and mimeType for downstream
    chunking and embedding.
    """
    creds = Credentials(token=access_token)
    drive = build("drive", "v3", credentials=creds)

    list_res = drive.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=100,
    ).execute()

    files = list_res.get("files", [])
    google_export_map = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain",
    }
    results = []

    for f in files:
        file_id = f.get("id")
        name = f.get("name")
        mime_type = f.get("mimeType")
        if not file_id or not name or not mime_type:
            continue
        try:
            export_mime = google_export_map.get(mime_type)
            if export_mime:
                content = (
                    drive.files()
                    .export(fileId=file_id, mimeType=export_mime)
                    .execute()
                )
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                results.append(
                    {"id": file_id, "name": name, "content": content, "mimeType": mime_type})
            elif mime_type.startswith("text/") or mime_type in (
                "application/json",
                "application/pdf",
            ):
                content = drive.files().get_media(fileId=file_id).execute()
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                results.append(
                    {"id": file_id, "name": name, "content": content, "mimeType": mime_type})
        except Exception as e:
            logging.exception("Skipping %s: %s", name, e)

    return results


def fetch_drive_file(file_id: str, access_token: str) -> list[dict]:
    """
    Fetch metadata and text content of a single Google Drive file by ID.

    Use case: When the user pastes a file URL (not a folder), we load only that file.
    Returns the same shape as fetch_drive_files: a list of one dict with id, name,
    content, mimeType. Unsupported mime types are skipped (empty list).
    """
    creds = Credentials(token=access_token)
    drive = build("drive", "v3", credentials=creds)

    try:
        meta = drive.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    except Exception as e:
        logging.exception("Failed to get file %s: %s", file_id, e)
        return []

    name = meta.get("name")
    mime_type = meta.get("mimeType")
    if not name or not mime_type:
        return []

    google_export_map = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain",
    }

    try:
        export_mime = google_export_map.get(mime_type)
        if export_mime:
            content = (
                drive.files()
                .export(fileId=file_id, mimeType=export_mime)
                .execute()
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return [{"id": file_id, "name": name, "content": content, "mimeType": mime_type}]
        if mime_type.startswith("text/") or mime_type in (
            "application/json",
            "application/pdf",
        ):
            content = drive.files().get_media(fileId=file_id).execute()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            return [{"id": file_id, "name": name, "content": content, "mimeType": mime_type}]
    except Exception as e:
        logging.exception("Skipping %s: %s", name, e)

    return []
