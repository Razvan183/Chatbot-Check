"""Load, chunk, and save the sample policy documents."""

from pathlib import Path
import sys
from collections.abc import Callable


# Allow both `python -m scripts.ingest_documents` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import DATA_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.db.database import SessionLocal, create_database_tables
from app.db.models import Document, DocumentChunk
from app.ingestion.chunking import build_chunk_key, chunk_text, extract_section_title
from app.ingestion.loaders import load_documents_from_folder


def ingest_documents(
    policy_folder: str | Path | None = None,
    session_factory: Callable[[], Session] = SessionLocal,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[int, int]:
    """Replace the demo corpus and return document and chunk totals."""
    source_folder = (
        Path(policy_folder)
        if policy_folder is not None
        else Path(DATA_DIR) / "sample_company_policy"
    )
    loaded_documents = load_documents_from_folder(source_folder)

    if not loaded_documents:
        raise ValueError(f"No supported documents found in: {source_folder}")

    create_database_tables()
    database_session = session_factory()
    total_chunks = 0

    try:
        # Chunks are deleted first because they reference their parent documents.
        database_session.execute(delete(DocumentChunk))
        database_session.execute(delete(Document))

        for loaded_document in loaded_documents:
            chunks = chunk_text(
                loaded_document["text"],
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            document = Document(
                filename=loaded_document["filename"],
                document_type=loaded_document["document_type"],
                source_path=loaded_document["source_path"],
                status="ready",
                num_chunks=len(chunks),
            )

            for chunk_index, chunk in enumerate(chunks):
                document.chunks.append(
                    DocumentChunk(
                        chunk_index=chunk_index,
                        chunk_key=build_chunk_key(
                            loaded_document["filename"],
                            chunk_index,
                        ),
                        chunk_text=chunk,
                        section_title=extract_section_title(chunk),
                    )
                )

            database_session.add(document)
            total_chunks += len(chunks)

        # One commit makes the replacement atomic: all changes succeed together.
        database_session.commit()
    except Exception:
        database_session.rollback()
        raise
    finally:
        database_session.close()

    return len(loaded_documents), total_chunks


def main() -> None:
    """Run demo ingestion and print its result."""
    document_count, chunk_count = ingest_documents()

    print(f"Loaded {document_count} documents")
    print(f"Created {chunk_count} chunks")
    print("Saved documents and chunks to database")


if __name__ == "__main__":
    main()
