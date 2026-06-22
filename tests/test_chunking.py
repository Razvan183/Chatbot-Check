"""Tests for paragraph-aware document chunking."""

import pytest

from app.ingestion.chunking import chunk_text, extract_section_title


def test_chunk_text_returns_non_empty_chunks() -> None:
    text = "# Policy\n\nFirst rule.\n\nSecond rule."

    chunks = chunk_text(text, chunk_size=25, overlap=5)

    assert isinstance(chunks, list)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_short_text_remains_one_chunk() -> None:
    text = "# Policy\n\nA short policy paragraph."

    assert chunk_text(text, chunk_size=100, overlap=20) == [text]


def test_overlap_repeats_trailing_paragraph_context() -> None:
    text = "First paragraph.\n\nShared paragraph.\n\nThird paragraph is longer."

    chunks = chunk_text(text, chunk_size=40, overlap=20)

    assert len(chunks) >= 2
    assert "paragraph." in chunks[0]
    assert "paragraph." in chunks[1]


def test_long_paragraph_is_split_near_target_size() -> None:
    text = "word " * 80

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert sum(chunk.split().count("word") for chunk in chunks) >= 80


def test_chunks_preserve_all_content_without_overlap() -> None:
    words = [f"word{index}" for index in range(80)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert " ".join(chunks).split() == words


def test_identical_neighboring_pieces_are_not_dropped() -> None:
    text = "word " * 80

    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert sum(len(chunk.split()) for chunk in chunks) == 80


def test_overlap_never_pushes_chunks_over_size_limit() -> None:
    text = f"{'A' * 48}\n\n{'B' * 48}\n\n{'C' * 48}"

    chunks = chunk_text(text, chunk_size=50, overlap=49)

    assert all(len(chunk) <= 50 for chunk in chunks)


def test_extract_section_title_returns_first_markdown_heading() -> None:
    chunk = "Introductory context.\n\n## Requesting Vacation\n\nSubmit a request."

    assert extract_section_title(chunk) == "Requesting Vacation"


def test_extract_section_title_returns_none_without_heading() -> None:
    assert extract_section_title("A paragraph without a heading.") is None


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_chunk_settings_raise_error(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("Policy text", chunk_size=chunk_size, overlap=overlap)
