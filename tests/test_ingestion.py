"""Tests for replacing source documents in the database."""

from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import Document, DocumentChunk
from scripts.ingest_documents import ingest_documents


def test_ingestion_replaces_existing_documents() -> None:
    policy_folder = Path(__file__).parent / "fixtures" / "ingestion_policies"
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    first_result = ingest_documents(
        policy_folder=policy_folder,
        session_factory=test_session_factory,
        chunk_size=100,
        chunk_overlap=20,
    )
    second_result = ingest_documents(
        policy_folder=policy_folder,
        session_factory=test_session_factory,
        chunk_size=100,
        chunk_overlap=20,
    )

    with test_session_factory() as session:
        document_count = session.scalar(select(func.count()).select_from(Document))
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk))
        orphan_count = session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(~DocumentChunk.document_id.in_(select(Document.id)))
        )

    assert first_result == second_result == (2, 2)
    assert document_count == 2
    assert chunk_count == 2
    assert orphan_count == 0
