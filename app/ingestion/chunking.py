"""Paragraph-aware text chunking for source documents."""

import re


PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n")
HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def build_chunk_key(filename: str, chunk_index: int) -> str:
    """Build a stable, readable key for a source chunk."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")

    return f"{filename.strip()}::{chunk_index}"


def extract_section_title(text_chunk: str) -> str | None:
    """Return the first Markdown heading found in a chunk."""
    match = HEADING_PATTERN.search(text_chunk)
    return match.group(1).strip() if match else None


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph without dropping any words."""
    words = paragraph.split()
    pieces: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        if len(word) > chunk_size:
            if current_words:
                pieces.append(" ".join(current_words))
                current_words = []
                current_length = 0

            pieces.extend(
                word[index : index + chunk_size]
                for index in range(0, len(word), chunk_size)
            )
            continue

        added_length = len(word) if not current_words else len(word) + 1
        if current_words and current_length + added_length > chunk_size:
            pieces.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += added_length

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def _prepare_paragraphs(text: str, chunk_size: int) -> list[str]:
    """Normalize paragraphs and split only those larger than the target."""
    paragraphs: list[str] = []

    for paragraph in PARAGRAPH_SEPARATOR.split(text.strip()):
        normalized = paragraph.strip()
        if not normalized:
            continue

        if len(normalized) <= chunk_size:
            paragraphs.append(normalized)
        else:
            paragraphs.extend(_split_long_paragraph(normalized, chunk_size))

    return paragraphs


def _build_base_chunks(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Pack every prepared paragraph exactly once into bounded chunks."""
    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join([*current, paragraph])
        if current and len(candidate) > chunk_size:
            chunks.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _take_overlap_suffix(text: str, maximum_length: int) -> str:
    """Take a readable suffix without returning a partial word."""
    if maximum_length <= 0 or not text:
        return ""
    if len(text) <= maximum_length:
        return text

    candidate = text[-maximum_length:]
    first_whitespace = re.search(r"\s", candidate)
    if first_whitespace is None:
        return ""

    return candidate[first_whitespace.end() :].strip()


def _add_overlap(
    base_chunks: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Prefix later chunks with as much prior context as safely fits."""
    if overlap == 0 or len(base_chunks) < 2:
        return base_chunks

    chunks = [base_chunks[0]]

    for index in range(1, len(base_chunks)):
        base_chunk = base_chunks[index]
        separator_length = 2
        available = chunk_size - len(base_chunk) - separator_length
        overlap_text = _take_overlap_suffix(
            base_chunks[index - 1],
            min(overlap, available),
        )

        if overlap_text:
            chunks.append(f"{overlap_text}\n\n{base_chunk}")
        else:
            chunks.append(base_chunk)

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into bounded chunks with best-effort trailing context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    paragraphs = _prepare_paragraphs(text, chunk_size)
    base_chunks = _build_base_chunks(paragraphs, chunk_size)
    return _add_overlap(base_chunks, chunk_size, overlap)
