"""Load the sample policy documents and print a short summary."""

from pathlib import Path
import sys


# Allow both `python -m scripts.inspect_documents` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATA_DIR
from app.ingestion.loaders import load_documents_from_folder


def main() -> None:
    """Print each loaded document's name, type, and text length."""
    policy_folder = f"{DATA_DIR}/sample_company_policy"
    documents = load_documents_from_folder(policy_folder)

    print(f"Loaded {len(documents)} documents from {policy_folder}")

    for document in documents:
        print(
            f"- {document['filename']} "
            f"({document['document_type']}, {len(document['text'])} characters)"
        )


if __name__ == "__main__":
    main()
