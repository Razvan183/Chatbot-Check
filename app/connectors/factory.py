"""Connector factory for selecting the RAG system under evaluation."""

from app.config import RAG_CONNECTOR
from app.connectors.base import RAGConnector, RAGConnectorError
from app.connectors.http_rag import HTTPRAGConnector
from app.connectors.internal_rag import InternalRAGConnector


def get_rag_connector(
    connector_name: str | None = None,
    http_url: str | None = None,
    timeout_seconds: float | None = None,
) -> RAGConnector:
    """Return the configured connector for evaluation runs."""
    normalized_name = (connector_name or RAG_CONNECTOR).strip().lower()
    if normalized_name == "internal":
        return InternalRAGConnector()
    if normalized_name == "http":
        return HTTPRAGConnector(
            endpoint_url=http_url if http_url is not None else None,
            timeout_seconds=timeout_seconds,
        )

    raise RAGConnectorError(
        "RAG_CONNECTOR must be one of: internal, http"
    )
