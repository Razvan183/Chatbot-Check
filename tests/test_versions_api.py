"""Tests for chatbot version API endpoints."""

from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.db.models import ChatbotVersion
from app.main import app


def test_list_chatbot_versions_is_ordered_by_name() -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        session.add_all(
            [
                ChatbotVersion(name="strict_refusal_v3", model_name="mock", top_k=5),
                ChatbotVersion(name="baseline_v1", model_name="mock", top_k=3),
            ]
        )
        session.commit()

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.get("/chatbot-versions")
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()

    assert response.status_code == 200
    payload = response.json()
    assert [version["name"] for version in payload] == [
        "baseline_v1",
        "strict_refusal_v3",
    ]
    assert payload[0]["top_k"] == 3


def test_create_chatbot_version_from_runtime_settings() -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chatbot-versions",
                json={
                    "name": "top_k_5_temp_0",
                    "description": "Tuned runtime version",
                    "model_name": "mock",
                    "top_k": 5,
                    "temperature": 0.0,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                },
            )
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "top_k_5_temp_0"
    assert payload["top_k"] == 5
    assert payload["temperature"] == 0.0


def test_create_chatbot_version_rejects_duplicate_name() -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        session.add(ChatbotVersion(name="baseline_v1", model_name="mock"))
        session.commit()

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chatbot-versions",
                json={"name": "baseline_v1", "model_name": "mock"},
            )
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()

    assert response.status_code == 409
    assert response.json() == {"detail": "Chatbot version name already exists"}
