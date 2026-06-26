"""Quality checks for the bundled HR policy evaluation dataset."""

import json
from pathlib import Path

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.ingestion.chunking import build_chunk_key, chunk_text
from app.ingestion.loaders import load_documents_from_folder


DATASET_PATH = Path("data") / "evaluation" / "hr_policy_eval_set.json"
POLICY_DIR = Path("data") / "sample_company_policy"


def _corpus_chunk_keys() -> set[str]:
    """Return stable chunk keys generated from the sample policy corpus."""
    keys: set[str] = set()
    for document in load_documents_from_folder(POLICY_DIR):
        chunks = chunk_text(
            document["text"],
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )
        keys.update(
            build_chunk_key(document["filename"], index)
            for index, _ in enumerate(chunks)
        )
    return keys


def test_hr_policy_eval_dataset_is_complete_and_grounded() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    case_ids = [case["id"] for case in cases]
    corpus_keys = _corpus_chunk_keys()

    assert dataset["name"] == "HR Policy Eval Set"
    assert len(cases) == 50
    assert len(set(case_ids)) == len(case_ids)

    question_types = {case["question_type"] for case in cases}
    assert {"factual", "citation_required", "multi_document", "misleading", "unanswerable"} <= question_types

    for case in cases:
        assert case["question"].strip()
        assert case["difficulty"] in {"easy", "medium", "hard"}
        assert isinstance(case["should_be_answerable"], bool)

        expected_keys = case["expected_chunk_keys"]
        assert set(expected_keys) <= corpus_keys

        if case["should_be_answerable"]:
            assert case["expected_answer"]
            assert expected_keys
        else:
            assert case["expected_answer"] is None
            assert expected_keys == []
