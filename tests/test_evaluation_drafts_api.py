"""Tests for draft evaluation dataset generation and review APIs."""

import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import Document, DocumentChunk, EvalCase, EvalDataset
from app.evaluation.dataset_generator import MockDatasetGenerator
from app.main import app


@pytest.fixture
def draft_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, sessionmaker], None, None]:
    """Provide an API client with chunks and a deterministic generator."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        document = Document(
            id=1,
            filename="policy.md",
            document_type="md",
            source_path="policy.md",
            status="ingested",
            num_chunks=2,
        )
        document.chunks = [
            DocumentChunk(
                id=1,
                chunk_index=0,
                chunk_key="policy.md::0",
                chunk_text="Employees receive 21 vacation days after two years.",
            ),
            DocumentChunk(
                id=2,
                chunk_index=1,
                chunk_key="policy.md::1",
                chunk_text="Gym memberships are not mentioned in this policy.",
            ),
        ]
        session.add(document)
        session.commit()

    generated = json.dumps(
        {
            "cases": [
                {
                    "id": "q001",
                    "question": "How many vacation days after two years?",
                    "expected_answer": "Employees receive 21 vacation days.",
                    "expected_chunk_keys": ["policy.md::0"],
                    "question_type": "factual",
                    "difficulty": "easy",
                    "should_be_answerable": True,
                    "confidence": 0.91,
                },
                {
                    "id": "q002",
                    "question": "Does the policy cover gym membership reimbursement?",
                    "expected_answer": None,
                    "expected_chunk_keys": [],
                    "question_type": "unanswerable",
                    "difficulty": "easy",
                    "should_be_answerable": False,
                    "confidence": 0.8,
                },
            ]
        }
    )
    monkeypatch.setattr(
        "app.evaluation.draft_generation.get_dataset_generator_provider",
        lambda: MockDatasetGenerator(generated),
    )

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client, test_session_factory
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()


def test_create_review_and_publish_draft_dataset(
    draft_client: tuple[TestClient, sessionmaker],
) -> None:
    client, session_factory = draft_client

    create_response = client.post(
        "/evaluation-drafts",
        json={
            "dataset_name": "Generated Policy Eval",
            "description": "Generated draft",
            "domain": "company_policy",
            "case_count": 2,
            "case_mix": {"factual": 1, "unanswerable": 1},
        },
    )

    assert create_response.status_code == 201
    draft_payload = create_response.json()
    assert draft_payload["name"] == "Generated Policy Eval"
    assert draft_payload["draft_case_count"] == 2
    assert draft_payload["approved_case_count"] == 0
    assert [case["status"] for case in draft_payload["cases"]] == ["draft", "draft"]

    first_case_id = draft_payload["cases"][0]["id"]
    second_case_id = draft_payload["cases"][1]["id"]

    approve_response = client.patch(
        f"/evaluation-drafts/{draft_payload['id']}/cases/{first_case_id}",
        json={"status": "approved", "reviewer_notes": "Grounded in chunk 0."},
    )
    reject_response = client.patch(
        f"/evaluation-drafts/{draft_payload['id']}/cases/{second_case_id}",
        json={"status": "rejected", "reviewer_notes": "Keep out of benchmark."},
    )

    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"
    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "rejected"

    publish_response = client.post(
        f"/evaluation-drafts/{draft_payload['id']}/publish"
    )

    assert publish_response.status_code == 200
    assert publish_response.json()["case_count"] == 1
    assert publish_response.json()["status"] == "published"

    with session_factory() as session:
        dataset_count = session.scalar(select(func.count()).select_from(EvalDataset))
        case_count = session.scalar(select(func.count()).select_from(EvalCase))
        published_case = session.scalar(select(EvalCase))

    assert dataset_count == 1
    assert case_count == 1
    assert published_case.question == "How many vacation days after two years?"
    assert json.loads(published_case.expected_chunk_keys) == ["policy.md::0"]


def test_publish_requires_at_least_one_approved_case(
    draft_client: tuple[TestClient, sessionmaker],
) -> None:
    client, _ = draft_client
    create_response = client.post(
        "/evaluation-drafts",
        json={
            "dataset_name": "Generated Policy Eval",
            "case_count": 2,
        },
    )
    draft_id = create_response.json()["id"]

    response = client.post(f"/evaluation-drafts/{draft_id}/publish")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "At least one draft case must be approved before publishing"
    }
