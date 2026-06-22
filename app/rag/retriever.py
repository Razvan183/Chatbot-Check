"""Semantic retrieval for ingested document chunks."""

from collections.abc import Callable
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk
from app.ingestion.embeddings import cosine_similarity, embed_texts


class RetrievedChunk(TypedDict):
    """Public shape returned for one ranked document chunk."""

    chunk_id: int
    document_id: int
    filename: str
    chunk_text: str
    score: float


def retrieve_chunks(
    question: str,
    top_k: int = 3,
    session_factory: Callable[[], Session] = SessionLocal,
) -> list[RetrievedChunk]:
    """Return the document chunks most semantically similar to a question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    with session_factory() as database_session:
        statement = (
            select(DocumentChunk, Document.filename)
            .join(Document, DocumentChunk.document_id == Document.id)
            .order_by(DocumentChunk.id)
        )
        chunk_rows = list(database_session.execute(statement))

    if not chunk_rows:
        return []

    chunk_texts = [chunk.chunk_text for chunk, _ in chunk_rows]
    vectors = embed_texts([question, *chunk_texts])
    question_vector = vectors[0]

    ranked_chunks: list[RetrievedChunk] = []
    for (chunk, filename), chunk_vector in zip(chunk_rows, vectors[1:], strict=True):
        ranked_chunks.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": filename,
                "chunk_text": chunk.chunk_text,
                "score": cosine_similarity(question_vector, chunk_vector),
            }
        )

    ranked_chunks.sort(key=lambda result: (-result["score"], result["chunk_id"]))
    return ranked_chunks[:top_k]
