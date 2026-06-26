"""Generate and persist draft evaluation datasets from ingested chunks."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, DraftEvalCase, DraftEvalDataset
from app.evaluation.dataset_generator import (
    DatasetGeneratorProvider,
    get_dataset_generator_provider,
)


ALLOWED_QUESTION_TYPES = {
    "factual",
    "citation_required",
    "multi_document",
    "misleading",
    "unanswerable",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


class DraftDatasetGenerationError(RuntimeError):
    """Raised when draft dataset generation cannot produce valid cases."""


def generate_draft_dataset(
    database_session: Session,
    dataset_name: str,
    description: str | None,
    domain: str,
    case_count: int,
    case_mix: dict[str, int] | None = None,
    provider: DatasetGeneratorProvider | None = None,
) -> DraftEvalDataset:
    """Generate candidate cases from chunks and persist them as draft rows."""
    if case_count <= 0:
        raise ValueError("case_count must be positive")

    chunk_payloads = _load_chunk_payloads(database_session)
    if not chunk_payloads:
        raise DraftDatasetGenerationError(
            "No ingested document chunks are available for dataset generation"
        )

    generator = provider or get_dataset_generator_provider()
    prompt = build_generation_prompt(
        dataset_name=dataset_name,
        description=description,
        domain=domain,
        case_count=case_count,
        case_mix=case_mix,
        chunks=chunk_payloads,
    )
    generated_text = generator.generate_text(prompt)
    candidate_cases = parse_generated_cases(
        generated_text,
        valid_chunk_keys={chunk["chunk_key"] for chunk in chunk_payloads},
    )
    if not candidate_cases:
        raise DraftDatasetGenerationError(
            "Dataset generator did not return any valid candidate cases"
        )

    draft_dataset = DraftEvalDataset(
        name=dataset_name.strip(),
        description=description.strip() if description else None,
        domain=domain.strip(),
        requested_case_count=case_count,
        status="draft",
    )
    database_session.add(draft_dataset)
    database_session.flush()

    for index, case in enumerate(candidate_cases[:case_count], start=1):
        database_session.add(
            DraftEvalCase(
                draft_dataset_id=draft_dataset.id,
                case_uid=str(case.get("id") or f"q{index:03d}"),
                question=case["question"],
                expected_answer=case.get("expected_answer"),
                expected_chunk_keys=json.dumps(case["expected_chunk_keys"]),
                question_type=case["question_type"],
                difficulty=case["difficulty"],
                should_be_answerable=case["should_be_answerable"],
                confidence=case.get("confidence"),
                status="draft",
            )
        )

    database_session.commit()
    database_session.refresh(draft_dataset)
    return draft_dataset


def _load_chunk_payloads(database_session: Session) -> list[dict[str, str]]:
    """Load chunk text and stable keys for prompt construction."""
    statement = (
        select(Document.filename, DocumentChunk.chunk_index, DocumentChunk.chunk_key, DocumentChunk.chunk_text)
        .join(Document, DocumentChunk.document_id == Document.id)
        .order_by(Document.filename, DocumentChunk.chunk_index)
    )
    payloads = []
    for filename, chunk_index, chunk_key, chunk_text in database_session.execute(statement):
        stable_key = chunk_key or f"{filename}::{chunk_index}"
        payloads.append(
            {
                "chunk_key": stable_key,
                "filename": filename,
                "chunk_text": chunk_text,
            }
        )
    return payloads


def build_generation_prompt(
    dataset_name: str,
    description: str | None,
    domain: str,
    case_count: int,
    case_mix: dict[str, int] | None,
    chunks: list[dict[str, str]],
) -> str:
    """Build a strict prompt for candidate eval-case generation."""
    chunk_blocks = "\n\n".join(
        f"[{chunk['chunk_key']}] {chunk['filename']}\n{chunk['chunk_text']}"
        for chunk in chunks
    )
    case_mix_text = json.dumps(case_mix or {}, ensure_ascii=True)

    return f"""You are creating draft evaluation cases for a RAG QA system.

Dataset name: {dataset_name}
Description: {description or ""}
Domain: {domain}
Requested cases: {case_count}
Requested case mix JSON: {case_mix_text}

Use only the source chunks below as evidence. Return JSON only, with this shape:
{{
  "cases": [
    {{
      "id": "q001",
      "question": "...",
      "expected_answer": "... or null",
      "expected_chunk_keys": ["filename.md::0"],
      "question_type": "factual|citation_required|multi_document|misleading|unanswerable",
      "difficulty": "easy|medium|hard",
      "should_be_answerable": true,
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Answerable cases must include expected_answer and at least one expected_chunk_key.
- Unanswerable cases must use expected_answer null and expected_chunk_keys [].
- expected_chunk_keys must exactly match chunk keys from the source chunks.
- Prefer realistic user questions and include numeric/date edge cases when available.
- These are draft cases for human review, not final ground truth.

Source chunks:
{chunk_blocks}
"""


def parse_generated_cases(
    generated_text: str,
    valid_chunk_keys: set[str],
) -> list[dict[str, Any]]:
    """Parse and validate generated JSON candidate cases."""
    try:
        parsed = json.loads(_strip_json_fence(generated_text))
    except json.JSONDecodeError as exc:
        raise DraftDatasetGenerationError("Dataset generator returned invalid JSON") from exc

    raw_cases = parsed.get("cases") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_cases, list):
        raise DraftDatasetGenerationError("Generated JSON must include a cases list")

    valid_cases = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            continue
        normalized = _normalize_case(raw_case, index, valid_chunk_keys)
        if normalized is not None:
            valid_cases.append(normalized)

    return valid_cases


def _strip_json_fence(text: str) -> str:
    """Remove a simple Markdown JSON fence if the model returned one."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _normalize_case(
    raw_case: dict[str, Any],
    index: int,
    valid_chunk_keys: set[str],
) -> dict[str, Any] | None:
    """Return a normalized candidate case or None if it is not reviewable."""
    question = raw_case.get("question")
    if not isinstance(question, str) or not question.strip():
        return None

    question_type = str(raw_case.get("question_type") or "factual").strip()
    if question_type not in ALLOWED_QUESTION_TYPES:
        question_type = "factual"

    difficulty = str(raw_case.get("difficulty") or "medium").strip()
    if difficulty not in ALLOWED_DIFFICULTIES:
        difficulty = "medium"

    should_be_answerable = bool(raw_case.get("should_be_answerable", True))
    raw_keys = raw_case.get("expected_chunk_keys") or []
    if not isinstance(raw_keys, list):
        raw_keys = []
    expected_chunk_keys = [
        str(key)
        for key in raw_keys
        if isinstance(key, str) and key in valid_chunk_keys
    ]

    expected_answer = raw_case.get("expected_answer")
    if expected_answer is not None:
        expected_answer = str(expected_answer).strip() or None

    if should_be_answerable and (not expected_answer or not expected_chunk_keys):
        return None
    if not should_be_answerable:
        expected_answer = None
        expected_chunk_keys = []
        question_type = "unanswerable"

    confidence = raw_case.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence = None

    return {
        "id": str(raw_case.get("id") or f"q{index:03d}"),
        "question": question.strip(),
        "expected_answer": expected_answer,
        "expected_chunk_keys": expected_chunk_keys,
        "question_type": question_type,
        "difficulty": difficulty,
        "should_be_answerable": should_be_answerable,
        "confidence": confidence,
    }
