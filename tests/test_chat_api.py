"""Tests for the chat API endpoint."""

from fastapi.testclient import TestClient
import pytest

from app.api import chat
from app.main import app
from app.rag.pipeline import ChatbotVersionNotFoundError


@pytest.fixture
def chat_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a test client with the RAG pipeline replaced by a fast double."""

    def fake_answer_question(question: str, chatbot_version_id: int) -> dict:
        return {
            "question": question,
            "answer": "Employees receive 21 vacation days after two years. [10]",
            "retrieved_chunks": [
                {
                    "chunk_id": 10,
                    "chunk_key": "vacation_policy.md::0",
                    "document_id": 2,
                    "filename": "vacation_policy.md",
                    "chunk_text": "Employees receive 21 vacation days after two years.",
                    "score": 0.95,
                }
            ],
            "citations": [10],
            "latency_ms": 12,
        }

    monkeypatch.setattr(chat, "answer_question", fake_answer_question)
    return TestClient(app)


def test_chat_endpoint_returns_pipeline_response(chat_client: TestClient) -> None:
    response = chat_client.post(
        "/chat",
        json={
            "question": "How many vacation days after two years?",
            "chatbot_version_id": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Employees receive 21 vacation days after two years. [10]"
    assert payload["citations"] == [10]
    assert payload["retrieved_chunks"][0]["chunk_id"] == 10
    assert payload["latency_ms"] == 12


def test_chat_endpoint_returns_404_for_missing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_: str, chatbot_version_id: int) -> dict:
        raise ChatbotVersionNotFoundError("Chatbot version not found")

    monkeypatch.setattr(chat, "answer_question", missing_version)

    response = TestClient(app).post(
        "/chat",
        json={
            "question": "What is the vacation policy?",
            "chatbot_version_id": 999,
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Chatbot version not found"}
