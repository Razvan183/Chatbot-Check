"""Connector for Chatbot Check's built-in demo RAG pipeline."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.connectors.base import RAGAnswer
from app.rag.pipeline import answer_question


class InternalRAGConnector:
    """Evaluate the internal RAG pipeline already provided by this app."""

    def answer(
        self,
        question: str,
        chatbot_version_id: int,
        session_factory: Callable[[], Session],
    ) -> RAGAnswer:
        """Return a normalized answer from the internal RAG pipeline."""
        result = answer_question(
            question,
            chatbot_version_id=chatbot_version_id,
            session_factory=session_factory,
        )
        return {
            "question": result["question"],
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "citations": result["citations"],
            "latency_ms": result["latency_ms"],
        }
