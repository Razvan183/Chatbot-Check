"""Tests for RAG connector abstractions."""

import httpx
import pytest

from app.connectors.base import FunctionRAGConnector, RAGConnectorError
from app.connectors.factory import get_rag_connector
from app.connectors.http_rag import HTTPRAGConnector
from app.connectors.internal_rag import InternalRAGConnector


def test_connector_factory_returns_internal_connector() -> None:
    connector = get_rag_connector("internal")

    assert isinstance(connector, InternalRAGConnector)


def test_connector_factory_rejects_unknown_connector() -> None:
    with pytest.raises(RAGConnectorError):
        get_rag_connector("unknown")


def test_function_connector_normalizes_legacy_answer_function() -> None:
    def fake_answer(question: str, chatbot_version_id: int, session_factory) -> dict:
        return {
            "question": question,
            "answer": "Answer [1]",
            "retrieved_chunks": [],
            "citations": [1],
            "latency_ms": 7,
        }

    connector = FunctionRAGConnector(fake_answer)

    assert connector.answer("Question?", 1, lambda: None) == {
        "question": "Question?",
        "answer": "Answer [1]",
        "retrieved_chunks": [],
        "citations": [1],
        "latency_ms": 7,
    }


def test_http_connector_normalizes_common_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
        assert url == "http://rag.local/chat"
        assert json == {"question": "What is the policy?", "chatbot_version_id": 3}
        assert timeout == 12
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "answer": "Use the approved system. [42]",
                "retrieved_chunks": [
                    {
                        "chunk_id": 42,
                        "chunk_key": "policy.md::0",
                        "document_id": 9,
                        "filename": "policy.md",
                        "chunk_text": "Use the approved system.",
                        "score": 0.91,
                    }
                ],
                "citations": [42],
                "latency_ms": 5,
            },
        )

    monkeypatch.setattr("app.connectors.http_rag.httpx.post", fake_post)
    connector = HTTPRAGConnector(
        endpoint_url="http://rag.local/chat",
        timeout_seconds=12,
    )

    result = connector.answer("What is the policy?", 3, lambda: None)

    assert result["answer"] == "Use the approved system. [42]"
    assert result["retrieved_chunks"][0]["chunk_key"] == "policy.md::0"
    assert result["citations"] == [42]
    assert result["latency_ms"] == 5
