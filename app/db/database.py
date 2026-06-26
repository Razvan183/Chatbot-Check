"""SQLAlchemy setup and database lifecycle helpers."""

from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    """Base class inherited by every SQLAlchemy model."""


# SQLite uses this option so FastAPI requests may share the same connection.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)


if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(Engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, _) -> None:
        """Enable SQLite foreign-key checks for every new connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and always close it after use."""
    database_session = SessionLocal()
    try:
        yield database_session
    finally:
        database_session.close()


def create_database_tables() -> None:
    """Create every table registered on the shared SQLAlchemy base."""
    # Importing the models registers their table metadata on Base.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()


def ensure_runtime_schema() -> None:
    """Apply tiny SQLite-safe schema additions used during early MVP work."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    inspector = inspect(engine)
    table_columns = {
        table_name: {
            column["name"] for column in inspector.get_columns(table_name)
        }
        for table_name in (
            "chat_logs",
            "documents",
            "document_chunks",
            "eval_cases",
            "eval_results",
        )
        if inspector.has_table(table_name)
    }

    required_columns_by_table = {
        "chat_logs": {
            "citations": "TEXT DEFAULT '[]'",
            "prompt": "TEXT",
            "settings_snapshot": "TEXT",
        },
        "document_chunks": {
            "chunk_key": "VARCHAR(300)",
        },
        "eval_cases": {
            "expected_chunk_keys": "TEXT DEFAULT '[]'",
        },
        "eval_results": {
            "retrieved_chunk_keys": "TEXT DEFAULT '[]'",
        },
    }

    with engine.begin() as connection:
        for table_name, required_columns in required_columns_by_table.items():
            existing_columns = table_columns.get(table_name)
            if existing_columns is None:
                continue

            for column_name, column_definition in required_columns.items():
                if column_name not in existing_columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN {column_name} {column_definition}"
                        )
                    )

        if "document_chunks" in table_columns and "documents" in table_columns:
            connection.execute(
                text(
                    """
                    UPDATE document_chunks
                    SET chunk_key = (
                        SELECT documents.filename || '::' || document_chunks.chunk_index
                        FROM documents
                        WHERE documents.id = document_chunks.document_id
                    )
                    WHERE chunk_key IS NULL OR chunk_key = ''
                    """
                )
            )
