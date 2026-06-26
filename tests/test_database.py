"""Tests for database connection behavior."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import Base, engine
from app.db.models import ChatLog, ChatbotVersion


def test_sqlite_foreign_keys_are_enabled() -> None:
    with engine.connect() as connection:
        enabled = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert enabled == 1


def test_create_database_tables_adds_chat_log_trace_columns(
    monkeypatch,
) -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_session_factory = sessionmaker(bind=test_engine)

    ChatbotVersion.__table__.create(test_engine)
    with test_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE chat_logs ("
                "id INTEGER PRIMARY KEY, "
                "chatbot_version_id INTEGER NOT NULL, "
                "question TEXT NOT NULL, "
                "answer TEXT NOT NULL, "
                "retrieved_chunk_ids TEXT NOT NULL DEFAULT '[]', "
                "latency_ms INTEGER, "
                "created_at DATETIME"
                ")"
            )
        )

    monkeypatch.setattr(database, "engine", test_engine)
    try:
        database.create_database_tables()

        with test_session_factory() as session:
            session.add(ChatbotVersion(id=1, name="baseline_v1"))
            session.add(
                ChatLog(
                    chatbot_version_id=1,
                    question="Question?",
                    answer="Answer.",
                    citations="[1]",
                    prompt="Prompt",
                    settings_snapshot='{"top_k": 3}',
                )
            )
            session.commit()

        with test_engine.connect() as connection:
            columns = {
                row[1]
                for row in connection.execute(text("PRAGMA table_info(chat_logs)"))
            }
    finally:
        test_engine.dispose()

    assert {"citations", "prompt", "settings_snapshot"} <= columns
