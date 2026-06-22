"""Read-only API endpoints for ingested documents."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Document, DocumentChunk
from app.db.schemas import DocumentChunkResponse, DocumentResponse


router = APIRouter(prefix="/documents", tags=["Documents"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=list[DocumentResponse])
def list_documents(database_session: DatabaseSession) -> list[Document]:
    """Return all ingested documents ordered by filename."""
    statement = select(Document).order_by(Document.filename)
    return list(database_session.scalars(statement))


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: int, database_session: DatabaseSession) -> Document:
    """Return one document or a clear 404 response."""
    document = database_session.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document


@router.get(
    "/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
)
def list_document_chunks(
    document_id: int,
    database_session: DatabaseSession,
) -> list[DocumentChunk]:
    """Return one document's chunks in their original order."""
    if database_session.get(Document, document_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(database_session.scalars(statement))
