"""Tests for the end-to-end RAG pipeline."""

from collections.abc import Callable
import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import ChatLog, ChatbotVersion
from app.rag import pipeline
from app.rag.pipeline import ChatbotVersionNotFoundError, answer_question


@pytest.fixture
def pipeline_session_factory() -> Callable[[], Session]:
    """Create an isolated database with one chatbot version."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        session.add(
            ChatbotVersion(
                id=1,
                name="baseline_v1",
                model_name="mock",
                top_k=2,
                temperature=0.2,
            )
        )
        session.add(
            ChatbotVersion(
                id=2,
                name="local_ollama_v1",
                model_name="qwen3:8b",
                top_k=4,
                temperature=0.1,
                prompt_template=(
                    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
                    "Use this refusal: {refusal_message}\n\nAnswer:"
                ),
            )
        )
        session.commit()

    try:
        yield test_session_factory
    finally:
        test_engine.dispose()


def test_answer_question_connects_retrieval_prompt_generation_and_logging(
    pipeline_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved_chunks = [
        {
            "chunk_id": 10,
            "chunk_key": "vacation_policy.md::0",
            "document_id": 2,
            "filename": "vacation_policy.md",
            "chunk_text": "Employees receive 21 vacation days after two years.",
            "score": 0.95,
        }
    ]
    calls = {}

    def fake_retrieve_chunks(question, top_k, session_factory):
        calls["retrieve"] = {
            "question": question,
            "top_k": top_k,
            "session_factory": session_factory,
        }
        return retrieved_chunks

    def fake_build_prompt(question, chunks, prompt_template=None):
        calls["prompt"] = {
            "question": question,
            "chunks": chunks,
            "prompt_template": prompt_template,
        }
        return "built prompt"

    def fake_generate_answer(
        prompt,
        generation_mode,
        model_name,
        temperature,
        question,
        retrieved_chunks,
    ):
        calls["generate"] = {
            "prompt": prompt,
            "generation_mode": generation_mode,
            "model_name": model_name,
            "temperature": temperature,
            "question": question,
            "retrieved_chunks": retrieved_chunks,
        }
        return "Employees receive 21 vacation days after two years. [10]"

    monkeypatch.setattr(pipeline, "retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr(pipeline, "build_rag_prompt", fake_build_prompt)
    monkeypatch.setattr(pipeline, "generate_answer", fake_generate_answer)

    result = answer_question(
        "How many vacation days after two years?",
        chatbot_version_id=1,
        session_factory=pipeline_session_factory,
    )

    assert calls["retrieve"]["top_k"] == 2
    assert calls["retrieve"]["session_factory"] is pipeline_session_factory
    assert calls["prompt"]["chunks"] == retrieved_chunks
    assert calls["prompt"]["prompt_template"] is None
    assert calls["generate"]["prompt"] == "built prompt"
    assert calls["generate"]["generation_mode"] == "mock"
    assert calls["generate"]["model_name"] is None
    assert calls["generate"]["temperature"] == 0.2
    assert calls["generate"]["retrieved_chunks"] == retrieved_chunks
    assert result["answer"] == "Employees receive 21 vacation days after two years. [10]"
    assert result["retrieved_chunks"] == retrieved_chunks
    assert result["citations"] == [10]
    assert result["latency_ms"] >= 0

    with pipeline_session_factory() as session:
        chat_log = session.scalar(select(ChatLog))

    assert chat_log is not None
    assert chat_log.chatbot_version_id == 1
    assert chat_log.question == "How many vacation days after two years?"
    assert chat_log.answer == result["answer"]
    assert json.loads(chat_log.retrieved_chunk_ids) == [10]
    assert json.loads(chat_log.citations) == [10]
    assert chat_log.prompt == "built prompt"
    settings_snapshot = json.loads(chat_log.settings_snapshot)
    assert settings_snapshot["name"] == "baseline_v1"
    assert settings_snapshot["generation_mode"] == "mock"
    assert settings_snapshot["top_k"] == 2


def test_answer_question_uses_version_prompt_and_ollama_settings(
    pipeline_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved_chunks = [
        {
            "chunk_id": 20,
            "chunk_key": "security_policy.md::0",
            "document_id": 3,
            "filename": "security_policy.md",
            "chunk_text": "Security incidents must be reported immediately.",
            "score": 0.9,
        }
    ]
    calls = {}

    def fake_retrieve_chunks(question, top_k, session_factory):
        calls["retrieve"] = {
            "question": question,
            "top_k": top_k,
            "session_factory": session_factory,
        }
        return retrieved_chunks

    monkeypatch.setattr(pipeline, "retrieve_chunks", fake_retrieve_chunks)

    def fake_build_prompt(question, chunks, prompt_template=None):
        calls["prompt_template"] = prompt_template
        return "custom built prompt"

    def fake_generate_answer(
        prompt,
        generation_mode,
        model_name,
        temperature,
        question,
        retrieved_chunks,
    ):
        calls["generate"] = {
            "prompt": prompt,
            "generation_mode": generation_mode,
            "model_name": model_name,
            "temperature": temperature,
        }
        return "Security incidents must be reported immediately. [20]"

    monkeypatch.setattr(pipeline, "build_rag_prompt", fake_build_prompt)
    monkeypatch.setattr(pipeline, "generate_answer", fake_generate_answer)

    result = answer_question(
        "When should security incidents be reported?",
        chatbot_version_id=2,
        session_factory=pipeline_session_factory,
    )

    assert calls["retrieve"]["top_k"] == 4
    assert "{context}" in calls["prompt_template"]
    assert calls["generate"] == {
        "prompt": "custom built prompt",
        "generation_mode": "ollama",
        "model_name": "qwen3:8b",
        "temperature": 0.1,
    }
    assert result["citations"] == [20]


def test_answer_question_rejects_missing_chatbot_version(
    pipeline_session_factory: Callable[[], Session],
) -> None:
    with pytest.raises(ChatbotVersionNotFoundError, match="Chatbot version not found"):
        answer_question(
            "What is the vacation policy?",
            chatbot_version_id=999,
            session_factory=pipeline_session_factory,
        )


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_answer_question_rejects_invalid_question(
    pipeline_session_factory: Callable[[], Session],
    question,
) -> None:
    with pytest.raises(ValueError, match="question"):
        answer_question(question, chatbot_version_id=1, session_factory=pipeline_session_factory)
