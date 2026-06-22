"""Create the demo chatbot configurations used by EvalForge."""

from collections.abc import Callable
from pathlib import Path
import sys


# Allow both module execution and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DEFAULT_EMBEDDING_MODEL
from app.db.database import SessionLocal, create_database_tables
from app.db.models import ChatbotVersion


DEMO_VERSIONS = (
    {
        "name": "baseline_v1",
        "description": "Baseline RAG configuration for comparison.",
        "model_name": "mock",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 3,
        "temperature": 0.2,
    },
    {
        "name": "more_context_v2",
        "description": "Retrieves more context while keeping baseline generation settings.",
        "model_name": "mock",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "temperature": 0.2,
    },
    {
        "name": "strict_refusal_v3",
        "description": "Uses more context and deterministic generation for stricter refusal.",
        "model_name": "mock",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "temperature": 0.0,
    },
    {
        "name": "weak_bad_demo_v4",
        "description": "Intentionally weak configuration used to demonstrate regression detection.",
        "model_name": "mock",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 1,
        "temperature": 0.7,
    },
)


def create_demo_versions(
    session_factory: Callable[[], Session] = SessionLocal,
) -> tuple[int, int]:
    """Create or update demo versions and return created and updated counts."""
    create_database_tables()
    created_count = 0
    updated_count = 0

    with session_factory() as database_session:
        try:
            for version_data in DEMO_VERSIONS:
                version = database_session.scalar(
                    select(ChatbotVersion).where(
                        ChatbotVersion.name == version_data["name"]
                    )
                )

                if version is None:
                    database_session.add(ChatbotVersion(**version_data))
                    created_count += 1
                    continue

                for field, value in version_data.items():
                    setattr(version, field, value)
                updated_count += 1

            database_session.commit()
        except Exception:
            database_session.rollback()
            raise

    return created_count, updated_count


def main() -> None:
    """Seed the demo chatbot versions and print a short summary."""
    created_count, updated_count = create_demo_versions()

    print(f"Created {created_count} chatbot versions")
    print(f"Updated {updated_count} chatbot versions")
    print(f"Database contains {len(DEMO_VERSIONS)} demo versions")


if __name__ == "__main__":
    main()
