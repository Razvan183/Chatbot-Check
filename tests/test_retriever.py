"""Tests for semantic document-chunk retrieval."""

from collections.abc import Callable

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import Document, DocumentChunk
from app.rag import retriever


@pytest.fixture
def retrieval_session_factory() -> Callable[[], Session]:
    """Create an isolated database containing a small retrieval corpus."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        vacation = Document(
            filename="vacation_policy.md",
            document_type="md",
            source_path="data/vacation_policy.md",
            status="ready",
            num_chunks=2,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    chunk_text="Employees receive 21 vacation days after two years.",
                ),
                DocumentChunk(
                    chunk_index=1,
                    chunk_text="Vacation requests require 10 business days notice.",
                ),
            ],
        )
        equipment = Document(
            filename="equipment_policy.md",
            document_type="md",
            source_path="data/equipment_policy.md",
            status="ready",
            num_chunks=1,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    chunk_text="Lost equipment must be reported within 24 hours.",
                )
            ],
        )
        session.add_all([vacation, equipment])
        session.commit()

    try:
        yield test_session_factory
    finally:
        test_engine.dispose()


def test_retrieve_chunks_returns_highest_scores_first(
    retrieval_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        assert texts[0] == "How many vacation days do employees receive?"
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ]

    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)

    results = retriever.retrieve_chunks(
        "How many vacation days do employees receive?",
        top_k=2,
        session_factory=retrieval_session_factory,
    )

    assert len(results) == 2
    assert [result["filename"] for result in results] == [
        "vacation_policy.md",
        "vacation_policy.md",
    ]
    assert "21 vacation days" in results[0]["chunk_text"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score"] >= results[1]["score"]
    assert set(results[0]) == {
        "chunk_id",
        "chunk_key",
        "document_id",
        "filename",
        "chunk_text",
        "score",
    }
    assert results[0]["chunk_key"] == "vacation_policy.md::0"


def test_retrieve_chunks_limits_results_to_available_chunks(
    retrieval_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    results = retriever.retrieve_chunks(
        "policy question",
        top_k=10,
        session_factory=retrieval_session_factory,
    )

    assert len(results) == 3
    assert [result["chunk_id"] for result in results] == sorted(
        result["chunk_id"] for result in results
    )


def test_retrieve_chunks_returns_empty_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    empty_session_factory = sessionmaker(bind=test_engine)

    def fail_if_called(_: list[str]) -> list[list[float]]:
        raise AssertionError("Embeddings should not be generated for an empty corpus")

    monkeypatch.setattr(retriever, "embed_texts", fail_if_called)

    try:
        assert (
            retriever.retrieve_chunks(
                "policy question",
                session_factory=empty_session_factory,
            )
            == []
        )
    finally:
        test_engine.dispose()


@pytest.mark.parametrize(
    ("question", "top_k"),
    [
        ("", 3),
        ("   ", 3),
        (123, 3),
        ("policy question", 0),
        ("policy question", -1),
        ("policy question", 1.5),
        ("policy question", True),
    ],
)
def test_retrieve_chunks_rejects_invalid_arguments(question, top_k) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve_chunks(question, top_k=top_k)
