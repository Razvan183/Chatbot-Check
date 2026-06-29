"""API endpoints for generating and reviewing draft evaluation datasets."""

from datetime import datetime, timezone
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import (
    Document,
    DocumentChunk,
    DraftEvalCase,
    DraftEvalDataset,
    EvalCase,
    EvalDataset,
)
from app.db.schemas import (
    DraftEvalCaseResponse,
    DraftEvalCaseReviewRequest,
    DraftEvalDatasetCreateRequest,
    DraftEvalDatasetPublishResponse,
    DraftEvalDatasetResponse,
    DraftEvalSupportingChunkResponse,
)
from app.evaluation.dataset_generator import DatasetGeneratorError
from app.evaluation.draft_generation import (
    DraftDatasetGenerationError,
    generate_draft_dataset,
)


router = APIRouter(prefix="/evaluation-drafts", tags=["Evaluation Drafts"])
DatabaseSession = Annotated[Session, Depends(get_db)]


def _decode_str_list(raw_value: str | None) -> list[str]:
    """Decode JSON string lists stored in draft case rows."""
    if raw_value is None:
        return []
    decoded = json.loads(raw_value)
    if not isinstance(decoded, list):
        return []
    return [str(value) for value in decoded]


def _supporting_chunks(
    database_session: Session,
    expected_chunk_keys: list[str],
) -> list[DraftEvalSupportingChunkResponse]:
    """Resolve expected chunk keys into source text for human review."""
    if not expected_chunk_keys:
        return []

    statement = (
        select(DocumentChunk, Document.filename)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(DocumentChunk.chunk_key.in_(expected_chunk_keys))
        .order_by(Document.filename, DocumentChunk.chunk_index)
    )
    chunks_by_key = {
        chunk.chunk_key: DraftEvalSupportingChunkResponse(
            chunk_key=chunk.chunk_key or f"{filename}::{chunk.chunk_index}",
            document_id=chunk.document_id,
            filename=filename,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
        )
        for chunk, filename in database_session.execute(statement)
    }
    return [
        chunks_by_key[chunk_key]
        for chunk_key in expected_chunk_keys
        if chunk_key in chunks_by_key
    ]


def _valid_chunk_keys(database_session: Session, chunk_keys: list[str]) -> list[str]:
    """Return the subset of chunk keys that exist in the current corpus."""
    if not chunk_keys:
        return []
    existing_keys = set(
        database_session.scalars(
            select(DocumentChunk.chunk_key).where(DocumentChunk.chunk_key.in_(chunk_keys))
        )
    )
    return [chunk_key for chunk_key in chunk_keys if chunk_key in existing_keys]


def _case_response(
    database_session: Session,
    draft_case: DraftEvalCase,
) -> DraftEvalCaseResponse:
    """Convert a draft case row into an API response."""
    expected_chunk_keys = _decode_str_list(draft_case.expected_chunk_keys)
    return DraftEvalCaseResponse(
        id=draft_case.id,
        case_uid=draft_case.case_uid,
        question=draft_case.question,
        expected_answer=draft_case.expected_answer,
        expected_chunk_keys=expected_chunk_keys,
        question_type=draft_case.question_type,
        difficulty=draft_case.difficulty,
        should_be_answerable=draft_case.should_be_answerable,
        confidence=draft_case.confidence,
        status=draft_case.status,
        reviewer_notes=draft_case.reviewer_notes,
        supporting_chunks=_supporting_chunks(database_session, expected_chunk_keys),
        created_at=draft_case.created_at,
        updated_at=draft_case.updated_at,
    )


def _draft_counts(database_session: Session, draft_dataset_id: int) -> dict[str, int]:
    """Return draft, approved, and rejected counts for a draft dataset."""
    statement = (
        select(DraftEvalCase.status, func.count(DraftEvalCase.id))
        .where(DraftEvalCase.draft_dataset_id == draft_dataset_id)
        .group_by(DraftEvalCase.status)
    )
    counts = {status_value: int(count) for status_value, count in database_session.execute(statement)}
    return {
        "draft": counts.get("draft", 0),
        "approved": counts.get("approved", 0),
        "rejected": counts.get("rejected", 0),
    }


def _draft_response(
    database_session: Session,
    draft_dataset: DraftEvalDataset,
) -> DraftEvalDatasetResponse:
    """Build a full draft dataset response."""
    counts = _draft_counts(database_session, draft_dataset.id)
    cases = list(
        database_session.scalars(
            select(DraftEvalCase)
            .where(DraftEvalCase.draft_dataset_id == draft_dataset.id)
            .order_by(DraftEvalCase.id)
        )
    )
    return DraftEvalDatasetResponse(
        id=draft_dataset.id,
        name=draft_dataset.name,
        description=draft_dataset.description,
        domain=draft_dataset.domain,
        status=draft_dataset.status,
        requested_case_count=draft_dataset.requested_case_count,
        draft_case_count=counts["draft"],
        approved_case_count=counts["approved"],
        rejected_case_count=counts["rejected"],
        published_eval_dataset_id=draft_dataset.published_eval_dataset_id,
        created_at=draft_dataset.created_at,
        cases=[_case_response(database_session, draft_case) for draft_case in cases],
    )


def _load_draft_or_404(
    database_session: Session,
    draft_dataset_id: int,
) -> DraftEvalDataset:
    """Load a draft dataset or raise a clear 404."""
    draft_dataset = database_session.get(DraftEvalDataset, draft_dataset_id)
    if draft_dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft evaluation dataset not found",
        )
    return draft_dataset


@router.post(
    "",
    response_model=DraftEvalDatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_draft_eval_dataset(
    request: DraftEvalDatasetCreateRequest,
    database_session: DatabaseSession,
) -> DraftEvalDatasetResponse:
    """Generate draft cases from ingested chunks using the authoring LLM."""
    try:
        draft_dataset = generate_draft_dataset(
            database_session=database_session,
            dataset_name=request.dataset_name,
            description=request.description,
            domain=request.domain,
            case_count=request.case_count,
            case_mix=request.case_mix,
        )
        return _draft_response(database_session, draft_dataset)
    except (DatasetGeneratorError, DraftDatasetGenerationError, ValueError) as exc:
        database_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[DraftEvalDatasetResponse])
def list_draft_eval_datasets(
    database_session: DatabaseSession,
) -> list[DraftEvalDatasetResponse]:
    """Return draft datasets ordered newest first."""
    statement = select(DraftEvalDataset).order_by(
        DraftEvalDataset.created_at.desc(),
        DraftEvalDataset.id.desc(),
    )
    return [
        _draft_response(database_session, draft_dataset)
        for draft_dataset in database_session.scalars(statement)
    ]


@router.get("/{draft_dataset_id}", response_model=DraftEvalDatasetResponse)
def get_draft_eval_dataset(
    draft_dataset_id: int,
    database_session: DatabaseSession,
) -> DraftEvalDatasetResponse:
    """Return one draft dataset and its candidate cases."""
    draft_dataset = _load_draft_or_404(database_session, draft_dataset_id)
    return _draft_response(database_session, draft_dataset)


@router.patch(
    "/{draft_dataset_id}/cases/{draft_case_id}",
    response_model=DraftEvalCaseResponse,
)
def review_draft_eval_case(
    draft_dataset_id: int,
    draft_case_id: int,
    request: DraftEvalCaseReviewRequest,
    database_session: DatabaseSession,
) -> DraftEvalCaseResponse:
    """Approve, reject, or edit a generated draft case."""
    _load_draft_or_404(database_session, draft_dataset_id)
    draft_case = database_session.get(DraftEvalCase, draft_case_id)
    if draft_case is None or draft_case.draft_dataset_id != draft_dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Draft evaluation case not found",
        )

    if request.question is not None:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="question cannot be empty",
            )
        draft_case.question = request.question.strip()
    if request.expected_answer is not None:
        draft_case.expected_answer = request.expected_answer.strip() or None
    if request.expected_chunk_keys is not None:
        draft_case.expected_chunk_keys = json.dumps(
            _valid_chunk_keys(database_session, request.expected_chunk_keys)
        )
    if request.question_type is not None:
        draft_case.question_type = request.question_type.strip()
    if request.difficulty is not None:
        draft_case.difficulty = request.difficulty.strip()
    if request.should_be_answerable is not None:
        draft_case.should_be_answerable = request.should_be_answerable
    if request.reviewer_notes is not None:
        draft_case.reviewer_notes = request.reviewer_notes.strip() or None

    draft_case.status = request.status
    draft_case.updated_at = datetime.now(timezone.utc)
    database_session.commit()
    database_session.refresh(draft_case)
    return _case_response(database_session, draft_case)


@router.post(
    "/{draft_dataset_id}/publish",
    response_model=DraftEvalDatasetPublishResponse,
)
def publish_draft_eval_dataset(
    draft_dataset_id: int,
    database_session: DatabaseSession,
) -> DraftEvalDatasetPublishResponse:
    """Publish approved draft cases into an official evaluation dataset."""
    draft_dataset = _load_draft_or_404(database_session, draft_dataset_id)
    if draft_dataset.published_eval_dataset_id is not None:
        return DraftEvalDatasetPublishResponse(
            eval_dataset_id=draft_dataset.published_eval_dataset_id,
            case_count=_approved_case_count(database_session, draft_dataset.id),
            status=draft_dataset.status,
        )

    approved_cases = list(
        database_session.scalars(
            select(DraftEvalCase)
            .where(DraftEvalCase.draft_dataset_id == draft_dataset_id)
            .where(DraftEvalCase.status == "approved")
            .order_by(DraftEvalCase.id)
        )
    )
    if not approved_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one draft case must be approved before publishing",
        )

    eval_dataset = EvalDataset(
        name=draft_dataset.name,
        description=draft_dataset.description,
        domain=draft_dataset.domain,
    )
    database_session.add(eval_dataset)
    database_session.flush()

    for draft_case in approved_cases:
        database_session.add(
            EvalCase(
                eval_dataset_id=eval_dataset.id,
                question=draft_case.question,
                expected_answer=draft_case.expected_answer,
                expected_chunk_ids=json.dumps([]),
                expected_chunk_keys=draft_case.expected_chunk_keys,
                question_type=draft_case.question_type,
                difficulty=draft_case.difficulty,
                should_be_answerable=draft_case.should_be_answerable,
            )
        )

    draft_dataset.status = "published"
    draft_dataset.published_eval_dataset_id = eval_dataset.id
    try:
        database_session.commit()
    except IntegrityError as exc:
        database_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An official evaluation dataset with this name already exists",
        ) from exc

    return DraftEvalDatasetPublishResponse(
        eval_dataset_id=eval_dataset.id,
        case_count=len(approved_cases),
        status="published",
    )


def _approved_case_count(database_session: Session, draft_dataset_id: int) -> int:
    """Return approved case count for one draft dataset."""
    return int(
        database_session.scalar(
            select(func.count(DraftEvalCase.id))
            .where(DraftEvalCase.draft_dataset_id == draft_dataset_id)
            .where(DraftEvalCase.status == "approved")
        )
        or 0
    )
