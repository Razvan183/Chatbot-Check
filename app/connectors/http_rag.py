"""HTTP connector for evaluating external RAG systems."""

from collections.abc import Callable
from time import perf_counter
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import HTTP_RAG_TIMEOUT_SECONDS, HTTP_RAG_URL
from app.connectors.base import RAGAnswer, RAGConnectorError, RetrievedContext


class HTTPRAGConnector:
    """Call an external RAG endpoint and normalize its response."""

    def __init__(
        self,
        endpoint_url: str = HTTP_RAG_URL,
        timeout_seconds: float = HTTP_RAG_TIMEOUT_SECONDS,
    ) -> None:
        if not endpoint_url.strip():
            raise RAGConnectorError("HTTP_RAG_URL must be configured for http connector")
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds

    def answer(
        self,
        question: str,
        chatbot_version_id: int,
        session_factory: Callable[[], Session],
    ) -> RAGAnswer:
        """Call the configured HTTP RAG endpoint."""
        del session_factory
        started_at = perf_counter()
        payload = {
            "question": question,
            "chatbot_version_id": chatbot_version_id,
        }

        try:
            response = httpx.post(
                self.endpoint_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RAGConnectorError(f"External RAG request failed: {exc}") from exc

        latency_ms = int((perf_counter() - started_at) * 1000)
        return _normalize_http_response(question, data, latency_ms)


def _normalize_http_response(
    question: str,
    data: dict[str, Any],
    latency_ms: int,
) -> RAGAnswer:
    """Normalize common RAG response shapes into Chatbot Check's answer shape."""
    if not isinstance(data, dict):
        raise RAGConnectorError("External RAG response must be a JSON object")

    raw_answer = data.get("answer") or data.get("response") or data.get("text")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise RAGConnectorError(
            "External RAG response must include a non-empty answer field"
        )

    raw_contexts = (
        data.get("retrieved_chunks")
        or data.get("contexts")
        or data.get("sources")
        or []
    )
    if not isinstance(raw_contexts, list):
        raise RAGConnectorError("External RAG contexts must be a list when provided")

    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = []

    return {
        "question": str(data.get("question") or question),
        "answer": raw_answer.strip(),
        "retrieved_chunks": [
            _normalize_context(index, raw_context)
            for index, raw_context in enumerate(raw_contexts, start=1)
        ],
        "citations": [int(citation) for citation in citations if _is_int_like(citation)],
        "latency_ms": int(data.get("latency_ms") or latency_ms),
    }


def _normalize_context(index: int, raw_context: Any) -> RetrievedContext:
    """Normalize one context object from an external RAG response."""
    if isinstance(raw_context, str):
        return {
            "chunk_id": index,
            "chunk_key": f"external::{index}",
            "document_id": 0,
            "filename": "external",
            "chunk_text": raw_context,
            "score": 0.0,
        }
    if not isinstance(raw_context, dict):
        raise RAGConnectorError("External RAG contexts must be strings or objects")

    chunk_id = raw_context.get("chunk_id") or raw_context.get("id") or index
    chunk_text = (
        raw_context.get("chunk_text")
        or raw_context.get("text")
        or raw_context.get("content")
        or ""
    )
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        raise RAGConnectorError("External RAG context objects must include text")

    return {
        "chunk_id": int(chunk_id) if _is_int_like(chunk_id) else index,
        "chunk_key": str(raw_context.get("chunk_key") or raw_context.get("key") or f"external::{index}"),
        "document_id": int(raw_context.get("document_id") or 0),
        "filename": str(raw_context.get("filename") or raw_context.get("source") or "external"),
        "chunk_text": chunk_text.strip(),
        "score": float(raw_context.get("score") or 0.0),
    }


def _is_int_like(value: Any) -> bool:
    """Return whether a value can safely become an integer."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
