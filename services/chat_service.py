"""
Chat business logic: retrieve context from vector DB and build prompt/messages for the LLM.
"""
from typing import Any, Protocol

from config import Config, get_config
from exceptions import ChatError


class EmbeddingsClientProtocol(Protocol):
    def embed(self, text: str) -> list[float]: ...


class VectorDBClientProtocol(Protocol):
    def search(
        self,
        vector: list[float],
        google_id: str,
        limit: int = 10,
        drive_url: str | None = None,
    ) -> list[dict[str, Any]]: ...


def _file_link(payload: dict[str, Any]) -> str:
    fid = payload.get("fileId")
    if fid:
        return f"https://drive.google.com/file/d/{fid}/view"
    return payload.get("driveUrl", "")


class ChatService:
    """Builds RAG context and chat message list. Dependencies are injectable for tests."""

    CHAT_SYSTEM_RULES = """Rules:
1. Base your answers ONLY on the provided document excerpts.
2. Put all citations at the end of your message. After your answer, add a "Sources:" section listing each source used as: [filename](link). If multiple files were used, list each one.
3. If information is not found in any of the excerpts, say so clearly.
4. Be concise but thorough. Use paragraphs for readability."""

    def __init__(
        self,
        embeddings_client: EmbeddingsClientProtocol,
        vector_db_client: VectorDBClientProtocol,
        config: Config | None = None,
    ) -> None:
        self._embeddings = embeddings_client
        self._vectordb = vector_db_client
        self._config = config or get_config()

    def get_system_prompt_and_messages(
        self,
        message: str,
        google_id: str,
        drive_url: str | None,
        history: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Embed the message, search vector DB, build context and system prompt, and
        return (system_prompt, chat_messages) for the LLM.
        """
        try:
            query_vector = self._embeddings.embed(message)
        except Exception as e:
            raise ChatError(str(e), user_message="Failed to process your message.") from e

        hits = self._vectordb.search(
            query_vector,
            google_id,
            limit=self._config.chat_search_limit,
            drive_url=drive_url,
        )

        context_parts = []
        for h in hits:
            payload = h.get("payload") or {}
            score = h.get("score", 0.0)
            name = payload.get("fileName", "unknown")
            link = _file_link(payload)
            content = (payload.get("content") or "").strip()
            context_parts.append(
                f"=== FILE: {name} (score: {score:.3f}) ===\nLink: {link}\n{content}\n=== END ==="
            )
        context = "\n\n".join(context_parts) if context_parts else "(No relevant documents found.)"

        system_prompt = f"""You are a knowledgeable assistant that answers questions about documents from a Google Drive folder.

Here are the most relevant document excerpts for the user's question:

{context}

{self.CHAT_SYSTEM_RULES}"""

        # Normalize history length and shape
        max_history = self._config.chat_max_history_messages
        normalized = []
        for m in (history or [])[:max_history]:
            role = (m.get("role") or "user").strip().lower()
            if role not in ("user", "assistant"):
                role = "user"
            normalized.append({"role": role, "content": (m.get("content") or "")})
        chat_messages = [*normalized, {"role": "user", "content": message}]

        return system_prompt, chat_messages
