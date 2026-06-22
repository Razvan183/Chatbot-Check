"""Tests for database connection behavior."""

from sqlalchemy import text

from app.db.database import engine


def test_sqlite_foreign_keys_are_enabled() -> None:
    with engine.connect() as connection:
        enabled = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert enabled == 1
