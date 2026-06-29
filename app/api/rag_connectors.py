"""API endpoints for configuring and testing the target RAG connector."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import HTTP_RAG_TIMEOUT_SECONDS, HTTP_RAG_URL, RAG_CONNECTOR
from app.connectors import RAGConnectorError, get_rag_connector
from app.db.database import get_db
from app.db.models import RAGConnectorConfig
from app.db.schemas import (
    RAGConnectorConfigRequest,
    RAGConnectorConfigResponse,
    RAGConnectorTestRequest,
    RAGConnectorTestResponse,
)


router = APIRouter(prefix="/rag-connector", tags=["RAG Connector"])
DatabaseSession = Annotated[Session, Depends(get_db)]


def _active_config(database_session: Session) -> RAGConnectorConfig | None:
    """Return the active persisted connector config, if one exists."""
    return database_session.scalar(
        select(RAGConnectorConfig)
        .where(RAGConnectorConfig.active.is_(True))
        .order_by(RAGConnectorConfig.id.desc())
    )


def _config_response(
    config: RAGConnectorConfig | None,
) -> RAGConnectorConfigResponse:
    """Build a public connector config response."""
    if config is None:
        return RAGConnectorConfigResponse(
            id=None,
            connector_type=RAG_CONNECTOR,
            http_url=HTTP_RAG_URL or None,
            timeout_seconds=HTTP_RAG_TIMEOUT_SECONDS,
            active=True,
            source="env",
        )

    return RAGConnectorConfigResponse(
        id=config.id,
        connector_type=config.connector_type,
        http_url=config.http_url,
        timeout_seconds=config.timeout_seconds,
        active=config.active,
        source="database",
    )


def _validate_request(request: RAGConnectorConfigRequest | RAGConnectorTestRequest) -> None:
    """Validate connector settings beyond Pydantic field types."""
    if request.connector_type == "http" and not (request.http_url or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="http_url is required when connector_type is http",
        )


@router.get("", response_model=RAGConnectorConfigResponse)
def get_rag_connector_config(
    database_session: DatabaseSession,
) -> RAGConnectorConfigResponse:
    """Return the active target RAG connector configuration."""
    return _config_response(_active_config(database_session))


@router.post("", response_model=RAGConnectorConfigResponse)
def save_rag_connector_config(
    request: RAGConnectorConfigRequest,
    database_session: DatabaseSession,
) -> RAGConnectorConfigResponse:
    """Persist and activate a target RAG connector configuration."""
    _validate_request(request)
    database_session.execute(
        update(RAGConnectorConfig).values(active=False)
    )
    config = RAGConnectorConfig(
        connector_type=request.connector_type,
        http_url=request.http_url.strip() if request.http_url else None,
        timeout_seconds=request.timeout_seconds,
        active=True,
    )
    database_session.add(config)
    database_session.commit()
    database_session.refresh(config)
    return _config_response(config)


@router.post("/test", response_model=RAGConnectorTestResponse)
def test_rag_connector(
    request: RAGConnectorTestRequest,
    database_session: DatabaseSession,
) -> RAGConnectorTestResponse:
    """Test connector settings by asking a simple question."""
    _validate_request(request)
    if request.connector_type == "internal":
        return RAGConnectorTestResponse(
            ok=True,
            message="Internal connector is available",
        )

    try:
        connector = get_rag_connector(
            connector_name=request.connector_type,
            http_url=request.http_url,
            timeout_seconds=request.timeout_seconds,
        )
        answer = connector.answer(
            request.question,
            chatbot_version_id=1,
            session_factory=lambda: database_session,
        )
    except (RAGConnectorError, ValueError) as exc:
        return RAGConnectorTestResponse(
            ok=False,
            message=str(exc),
        )

    return RAGConnectorTestResponse(
        ok=True,
        message="Connector test succeeded",
        answer_preview=answer["answer"][:240],
        retrieved_chunk_count=len(answer["retrieved_chunks"]),
        latency_ms=answer["latency_ms"],
    )
