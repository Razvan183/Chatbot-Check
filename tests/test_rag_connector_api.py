"""Tests for RAG connector configuration API."""

from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import app


def test_get_save_and_test_rag_connector_config() -> None:
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
            default_response = client.get("/rag-connector")
            save_response = client.post(
                "/rag-connector",
                json={
                    "connector_type": "http",
                    "http_url": "http://rag.local/chat",
                    "timeout_seconds": 12,
                },
            )
            saved_response = client.get("/rag-connector")
            test_response = client.post(
                "/rag-connector/test",
                json={"connector_type": "internal"},
            )
            missing_url_response = client.post(
                "/rag-connector",
                json={"connector_type": "http"},
            )
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()

    assert default_response.status_code == 200
    assert default_response.json()["source"] == "env"
    assert save_response.status_code == 200
    assert save_response.json()["source"] == "database"
    assert save_response.json()["connector_type"] == "http"
    assert save_response.json()["http_url"] == "http://rag.local/chat"
    assert saved_response.json()["timeout_seconds"] == 12
    assert test_response.status_code == 200
    assert test_response.json()["ok"] is True
    assert missing_url_response.status_code == 400
    assert missing_url_response.json() == {
        "detail": "http_url is required when connector_type is http"
    }
