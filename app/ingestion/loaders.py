"""Functions for loading source documents from disk."""

from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file and return its complete contents."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    return file_path.read_text(encoding="utf-8")


def load_markdown_file(path: str | Path) -> str:
    """Read a Markdown file and return its source text unchanged."""
    file_path = Path(path)

    if file_path.suffix.lower() != ".md":
        raise ValueError(f"Expected a Markdown file, received: {file_path}")

    return load_text_file(file_path)


def load_documents_from_folder(folder_path: str | Path) -> list[dict[str, str]]:
    """Load supported documents from a folder in alphabetical order."""
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"Document folder not found: {folder}")

    documents: list[dict[str, str]] = []

    for file_path in sorted(folder.iterdir(), key=lambda path: path.name.lower()):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        if extension == ".md":
            text = load_markdown_file(file_path)
        else:
            text = load_text_file(file_path)

        documents.append(
            {
                "filename": file_path.name,
                "source_path": str(file_path),
                "document_type": extension.removeprefix("."),
                "text": text,
            }
        )

    return documents
