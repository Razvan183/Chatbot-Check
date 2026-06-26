"""Shared connector contract for RAG systems under evaluation."""

from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from sqlalchemy.orm import Session


class RetrievedContext(TypedDict, total=False):
    """Normalized retrieved context returned by a RAG connector."""

    chunk_id: int
    chunk_key: str
    document_id: int
    filename: str
    chunk_text: str
    score: float


class RAGAnswer(TypedDict):
    """Normalized answer returned by a RAG connector."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedContext]
    citations: list[int]
    latency_ms: int


class RAGConnector(Protocol):
    """Interface implemented by every RAG target connector."""

    def answer(
        self,
        question: str,
        chatbot_version_id: int,
        session_factory: Callable[[], Session],
    ) -> RAGAnswer:
        """Answer a question using the target RAG system."""


class RAGConnectorError(RuntimeError):
    """Raised when a connector cannot call or normalize a RAG response."""


class FunctionRAGConnector:
    """Adapter for legacy answer functions and tests."""

    def __init__(self, answer_function: Callable[..., dict[str, Any]]) -> None:
        self.answer_function = answer_function

    def answer(
        self,
        question: str,
        chatbot_version_id: int,
        session_factory: Callable[[], Session],
    ) -> RAGAnswer:
        """Call a function with the historical answer_question signature."""
        result = self.answer_function(
            question,
            chatbot_version_id=chatbot_version_id,
            session_factory=session_factory,
        )
        return {
            "question": str(result.get("question", question)),
            "answer": str(result.get("answer", "")),
            "retrieved_chunks": list(result.get("retrieved_chunks", [])),
            "citations": list(result.get("citations", [])),
            "latency_ms": int(result.get("latency_ms", 0)),
        }
