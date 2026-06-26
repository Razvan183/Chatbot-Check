"""End-to-end RAG question answering pipeline."""

import json
import re
from time import perf_counter
from typing import Any, TypedDict

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import ChatLog, ChatbotVersion
from app.rag.generator import generate_answer
from app.rag.prompts import build_rag_prompt
from app.rag.retriever import RetrievedChunk, retrieve_chunks


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class ChatbotVersionNotFoundError(ValueError):
    """Raised when a requested chatbot version does not exist."""


class AnswerResult(TypedDict):
    """Public result returned by the RAG pipeline."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    citations: list[int]
    latency_ms: int


class VersionSettings(TypedDict):
    """Runtime settings copied from a chatbot version."""

    chatbot_version_id: int
    name: str
    model_name: str
    generation_mode: str
    generation_model: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    prompt_template: str | None


def _extract_citations(answer: str) -> list[int]:
    """Return unique cited chunk IDs in the order they appear."""
    citations: list[int] = []
    seen: set[int] = set()

    for match in CITATION_PATTERN.finditer(answer):
        chunk_id = int(match.group(1))
        if chunk_id not in seen:
            citations.append(chunk_id)
            seen.add(chunk_id)

    return citations


def _load_chatbot_version(
    database_session: Session,
    chatbot_version_id: int,
) -> ChatbotVersion:
    """Load the chatbot version or raise a clear domain error."""
    if (
        not isinstance(chatbot_version_id, int)
        or isinstance(chatbot_version_id, bool)
        or chatbot_version_id <= 0
    ):
        raise ValueError("chatbot_version_id must be a positive integer")

    chatbot_version = database_session.get(ChatbotVersion, chatbot_version_id)
    if chatbot_version is None:
        raise ChatbotVersionNotFoundError("Chatbot version not found")

    return chatbot_version


def _generation_settings(model_name: str) -> tuple[str, str | None]:
    """Translate stored version model names into a generation backend."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("chatbot version model_name must be a non-empty string")

    normalized_model = model_name.strip()
    if normalized_model.lower() == "mock":
        return "mock", None

    return "ollama", normalized_model


def _snapshot_version_settings(chatbot_version: ChatbotVersion) -> VersionSettings:
    """Copy version settings so logs keep the exact runtime configuration."""
    generation_mode, generation_model = _generation_settings(chatbot_version.model_name)

    return {
        "chatbot_version_id": chatbot_version.id,
        "name": chatbot_version.name,
        "model_name": chatbot_version.model_name,
        "generation_mode": generation_mode,
        "generation_model": generation_model,
        "embedding_model": chatbot_version.embedding_model,
        "chunk_size": chatbot_version.chunk_size,
        "chunk_overlap": chatbot_version.chunk_overlap,
        "top_k": chatbot_version.top_k,
        "temperature": chatbot_version.temperature,
        "prompt_template": chatbot_version.prompt_template,
    }


def answer_question(
    question: str,
    chatbot_version_id: int,
    session_factory: Any = SessionLocal,
) -> AnswerResult:
    """Answer one question, persist the chat log, and return trace details."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    started_at = perf_counter()

    with session_factory() as database_session:
        chatbot_version = _load_chatbot_version(database_session, chatbot_version_id)
        version_settings = _snapshot_version_settings(chatbot_version)

    retrieved_chunks = retrieve_chunks(
        question,
        top_k=version_settings["top_k"],
        session_factory=session_factory,
    )
    prompt = build_rag_prompt(
        question,
        retrieved_chunks,
        prompt_template=version_settings["prompt_template"],
    )
    answer = generate_answer(
        prompt,
        generation_mode=version_settings["generation_mode"],
        model_name=version_settings["generation_model"],
        temperature=version_settings["temperature"],
        question=question,
        retrieved_chunks=retrieved_chunks,
    )
    latency_ms = int((perf_counter() - started_at) * 1000)
    citations = _extract_citations(answer)

    retrieved_chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]

    with session_factory() as database_session:
        database_session.add(
            ChatLog(
                chatbot_version_id=chatbot_version_id,
                question=question.strip(),
                answer=answer,
                retrieved_chunk_ids=json.dumps(retrieved_chunk_ids),
                citations=json.dumps(citations),
                prompt=prompt,
                settings_snapshot=json.dumps(version_settings),
                latency_ms=latency_ms,
            )
        )
        database_session.commit()

    return {
        "question": question.strip(),
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations,
        "latency_ms": latency_ms,
    }
