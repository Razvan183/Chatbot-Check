"""Tests for loading source documents from disk."""

from pathlib import Path

import pytest

from app.ingestion.loaders import (
    load_documents_from_folder,
    load_markdown_file,
    load_text_file,
)


FIXTURE_FOLDER = Path(__file__).parent / "fixtures" / "loader_documents"


def test_load_text_file_preserves_utf8_content() -> None:
    text = load_text_file(FIXTURE_FOLDER / "a_policy.md")

    assert "€150" in text


def test_load_markdown_rejects_non_markdown_file() -> None:
    with pytest.raises(ValueError):
        load_markdown_file(FIXTURE_FOLDER / "b_policy.txt")


def test_folder_loader_is_ordered_and_ignores_unsupported_files() -> None:
    documents = load_documents_from_folder(FIXTURE_FOLDER)

    assert [document["filename"] for document in documents] == [
        "a_policy.md",
        "b_policy.txt",
    ]
    assert [document["document_type"] for document in documents] == ["md", "txt"]


def test_folder_loader_rejects_missing_folder() -> None:
    with pytest.raises(NotADirectoryError):
        load_documents_from_folder(FIXTURE_FOLDER / "missing")
