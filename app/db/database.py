"""SQLAlchemy setup and database lifecycle helpers."""

from collections.abc import Generator

from sqlalchemy import create_engine, event
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
