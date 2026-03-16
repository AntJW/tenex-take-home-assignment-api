import uuid
import json
import logging

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

from clients.embeddings_client import EmbeddingsAPIClient
from clients.vector_db_client import VectorDBClient
from clients.llm_client import LLMClient
from utils import parse_drive_url, chunk_text, fetch_drive_files, fetch_drive_file

load_dotenv()

app = Flask(__name__)

CORS(app)

embeddings = EmbeddingsAPIClient()
vectordb = VectorDBClient()
llm = LLMClient()


@app.post("/api/drive/load")
def api_drive_load():
    data = request.get_json() or {}
    drive_url = data.get("driveUrl") or data.get("drive_url")
    access_token = data.get("accessToken") or data.get("access_token")
    google_id = data.get("googleId") or data.get("google_id")

    if not drive_url or not access_token:
        return jsonify({"error": "driveUrl and accessToken are required"}), 400
    if not google_id:
        return jsonify({"error": "googleId is required"}), 400

    parsed = parse_drive_url(drive_url)
    if not parsed:
        logging.warning("Invalid drive URL: %r",
                        drive_url[:200] if drive_url else None)
        return jsonify({"error": "Invalid Google Drive folder or file URL"}), 400

    url_type, resource_id = parsed
    try:
        if url_type == "folder":
            files = fetch_drive_files(resource_id, access_token)
        else:
            files = fetch_drive_file(resource_id, access_token)

        vectordb.ensure_collection()
        vectordb.delete_by_drive_url(google_id, drive_url)

        all_chunks: list[dict] = []
        for file in files:
            for text in chunk_text(file["content"]):
                all_chunks.append({
                    "text": text,
                    "fileName": file["name"],
                    "fileId": file.get("id"),
                })

        if all_chunks:
            vectors = embeddings.embed_batch([c["text"] for c in all_chunks])
            points = [
                {
                    "id": str(uuid.uuid4()),
                    "vector": vectors[i],
                    "payload": {
                        "googleId": google_id,
                        "fileName": all_chunks[i]["fileName"],
                        "fileId": all_chunks[i].get("fileId"),
                        "driveUrl": drive_url,
                        "content": all_chunks[i]["text"],
                    },
                }
                for i in range(len(all_chunks))
            ]
            BATCH_SIZE = 100
            for i in range(0, len(points), BATCH_SIZE):
                vectordb.upsert(points[i: i + BATCH_SIZE])

        logging.info(
            "Indexed %s chunks from %s files for user %s",
            len(all_chunks),
            len(files),
            google_id,
        )

        return jsonify(
            {
                "folderId": resource_id,
                "files": [{"name": f["name"], "mimeType": f["mimeType"]} for f in files],
                "chunksIndexed": len(all_chunks),
            }
        )
    except Exception as e:
        logging.exception("Drive load error: %s", e)
        return (
            jsonify(
                {"error": "Failed to load Google Drive folder or file. Check your permissions."}
            ),
            500,
        )


@app.post("/api/agent/chat")
def api_agent_chat():
    data = request.get_json() or {}
    message = data.get("message")
    google_id = data.get("googleId")
    history = data.get("history") or []

    if not message or not google_id:
        return jsonify({"error": "message and googleId are required"}), 400

    try:
        query_vector = embeddings.embed(message)
        hits = vectordb.search(query_vector, google_id, 10)

        def file_link(payload: dict) -> str:
            fid = payload.get("fileId")
            return f"https://drive.google.com/file/d/{fid}/view" if fid else payload.get("driveUrl", "")

        context = "\n\n".join(
            f"=== FILE: {h['payload']['fileName']} (score: {h['score']:.3f}) ===\nLink: {file_link(h['payload'])}\n{h['payload']['content']}\n=== END ==="
            for h in hits
        ) or "(No relevant documents found.)"

        system_prompt = f"""You are a knowledgeable assistant that answers questions about documents from a Google Drive folder.

Here are the most relevant document excerpts for the user's question:

{context}

Rules:
1. Base your answers ONLY on the provided document excerpts.
2. After each claim or piece of information, include a citation with the filename and the file link: [Source: filename](link)
3. If multiple files support a point, cite all relevant files with their links.
4. If information is not found in any of the excerpts, say so clearly.
5. Be concise but thorough. Use paragraphs for readability."""

        chat_messages = [
            *[{"role": m.get("role", "user"), "content": m.get("content", "")}
              for m in history],
            {"role": "user", "content": message},
        ]
    except Exception as e:
        logging.exception("Chat error: %s", e)
        return jsonify({"error": str(e)}), 500

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
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
