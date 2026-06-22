"""Tests for the document API endpoints."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import Document, DocumentChunk
from app.main import app


@pytest.fixture
def document_client() -> Generator[TestClient, None, None]:
    """Provide an API client backed by an isolated SQLite database."""
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
        )
        vacation.chunks = [
            DocumentChunk(
                chunk_index=1,
                chunk_text="Second chunk",
                section_title="Requests",
            ),
            DocumentChunk(
                chunk_index=0,
                chunk_text="First chunk",
                section_title="Vacation Policy",
            ),
        ]
        session.add(vacation)
        session.add(
            Document(
                filename="equipment_policy.md",
                document_type="md",
                source_path="data/equipment_policy.md",
                status="ready",
                num_chunks=0,
            )
        )
        session.commit()

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()


def test_list_documents_is_ordered_by_filename(document_client: TestClient) -> None:
    response = document_client.get("/documents")

    assert response.status_code == 200
    payload = response.json()
    assert [document["filename"] for document in payload] == [
        "equipment_policy.md",
        "vacation_policy.md",
    ]


def test_get_document_returns_metadata(document_client: TestClient) -> None:
    documents = document_client.get("/documents").json()
    vacation = next(
        document
        for document in documents
        if document["filename"] == "vacation_policy.md"
    )

    response = document_client.get(f"/documents/{vacation['id']}")

    assert response.status_code == 200
    assert response.json()["num_chunks"] == 2


def test_list_chunks_is_ordered_by_chunk_index(
    document_client: TestClient,
) -> None:
    documents = document_client.get("/documents").json()
    vacation_id = next(
        document["id"]
        for document in documents
        if document["filename"] == "vacation_policy.md"
    )

    response = document_client.get(f"/documents/{vacation_id}/chunks")

    assert response.status_code == 200
    payload = response.json()
    assert [chunk["chunk_index"] for chunk in payload] == [0, 1]
    assert [chunk["chunk_text"] for chunk in payload] == [
        "First chunk",
        "Second chunk",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/documents/999999",
        "/documents/999999/chunks",
    ],
)
def test_missing_document_returns_404(
    document_client: TestClient,
    path: str,
) -> None:
    response = document_client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
