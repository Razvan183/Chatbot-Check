"""Run evaluation datasets against chatbot versions."""

from collections.abc import Callable
from datetime import datetime, timezone
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors import RAGConnector, get_rag_connector
from app.connectors.base import FunctionRAGConnector
from app.db.database import SessionLocal, create_database_tables
from app.db.models import (
    ChatbotVersion,
    EvalCase,
    EvalDataset,
    EvalResult,
    EvalRun,
    RAGConnectorConfig,
)
from app.evaluation.failure_classifier import classify_failure
from app.evaluation.metrics import (
    answer_keyword_score,
    citation_presence,
    expected_chunk_hit,
    expected_chunk_key_hit,
    numeric_consistency,
    refusal_correctness,
)
from app.evaluation.scoring import calculate_case_score, has_hallucination_flag


class EvaluationDatasetNotFoundError(ValueError):
    """Raised when an evaluation dataset does not exist."""


class EvaluationRunError(RuntimeError):
    """Raised when an evaluation run cannot be completed."""


def _active_connector_settings(database_session: Session) -> dict[str, Any]:
    """Load the active persisted connector settings, if configured."""
    config = database_session.scalar(
        select(RAGConnectorConfig)
        .where(RAGConnectorConfig.active.is_(True))
        .order_by(RAGConnectorConfig.id.desc())
    )
    if config is None:
        return {}
    return {
        "connector_name": config.connector_type,
        "http_url": config.http_url,
        "timeout_seconds": config.timeout_seconds,
    }


def _load_dataset_cases(
    database_session: Session,
    eval_dataset_id: int,
) -> list[dict[str, Any]]:
    """Load eval cases or raise a clear error."""
    dataset = database_session.get(EvalDataset, eval_dataset_id)
    if dataset is None:
        raise EvaluationDatasetNotFoundError("Evaluation dataset not found")

    statement = (
        select(EvalCase)
        .where(EvalCase.eval_dataset_id == eval_dataset_id)
        .order_by(EvalCase.id)
    )
    cases = list(database_session.scalars(statement))
    if not cases:
        raise EvaluationRunError("Evaluation dataset has no cases")

    return [_case_to_dict(eval_case) for eval_case in cases]


def _validate_chatbot_version(
    database_session: Session,
    chatbot_version_id: int,
) -> None:
    """Raise a clear error when the chatbot version does not exist."""
    if database_session.get(ChatbotVersion, chatbot_version_id) is None:
        from app.rag.pipeline import ChatbotVersionNotFoundError

        raise ChatbotVersionNotFoundError("Chatbot version not found")


def _case_to_dict(eval_case: EvalCase) -> dict[str, Any]:
    """Convert an ORM eval case into the shape used by the classifier."""
    return {
        "id": eval_case.id,
        "question": eval_case.question,
        "expected_answer": eval_case.expected_answer,
        "expected_chunk_ids": json.loads(eval_case.expected_chunk_ids or "[]"),
        "expected_chunk_keys": json.loads(eval_case.expected_chunk_keys or "[]"),
        "question_type": eval_case.question_type,
        "difficulty": eval_case.difficulty,
        "should_be_answerable": eval_case.should_be_answerable,
    }


def _retrieved_context(answer_result: dict) -> str:
    """Join retrieved chunk text into one context string for metric checks."""
    return "\n\n".join(
        chunk["chunk_text"] for chunk in answer_result["retrieved_chunks"]
    )


def _retrieved_chunk_ids(answer_result: dict) -> list[int]:
    """Extract retrieved chunk IDs from an answer result."""
    return [chunk["chunk_id"] for chunk in answer_result["retrieved_chunks"]]


def _retrieved_chunk_keys(answer_result: dict) -> list[str]:
    """Extract stable retrieved chunk keys from an answer result."""
    return [
        str(chunk["chunk_key"])
        for chunk in answer_result["retrieved_chunks"]
        if chunk.get("chunk_key")
    ]


def evaluate_case(case: dict[str, Any], answer_result: dict) -> dict[str, Any]:
    """Compute metrics, score, and failure classification for one case."""
    answer = answer_result["answer"]
    retrieved_ids = _retrieved_chunk_ids(answer_result)
    retrieved_keys = _retrieved_chunk_keys(answer_result)
    context = _retrieved_context(answer_result)
    expected_keys = case.get("expected_chunk_keys", [])

    retrieval_hit = (
        expected_chunk_key_hit(expected_keys, retrieved_keys)
        if expected_keys
        else expected_chunk_hit(case["expected_chunk_ids"], retrieved_ids)
    )

    metrics = {
        "expected_chunk_hit": retrieval_hit,
        "citation_presence": citation_presence(answer),
        "refusal_score": refusal_correctness(answer, case["should_be_answerable"]),
        "numeric_consistency": numeric_consistency(answer, context),
        "answer_keyword_score": answer_keyword_score(
            case["expected_answer"],
            answer,
        ),
    }
    metrics["hallucination_flag"] = has_hallucination_flag(
        metrics,
        case["should_be_answerable"],
        case["question_type"],
    )
    score = calculate_case_score(
        metrics,
        case["should_be_answerable"],
        case["question_type"],
    )
    failure_type, failure_reason = classify_failure(case, metrics, score)

    return {
        "generated_answer": answer,
        "retrieved_chunk_ids": retrieved_ids,
        "retrieved_chunk_keys": retrieved_keys,
        "metrics": metrics,
        "overall_case_score": score,
        "passed": failure_type == "passed",
        "failure_type": failure_type,
        "failure_reason": failure_reason,
    }


def run_evaluation(
    eval_dataset_id: int,
    chatbot_version_id: int,
    run_name: str,
    session_factory: Callable[[], Session] = SessionLocal,
    answer_function: Callable[..., dict] | None = None,
    connector: RAGConnector | None = None,
) -> dict[str, Any]:
    """Run every case in an eval dataset and persist the results."""
    eval_run_id = create_evaluation_run_record(
        eval_dataset_id=eval_dataset_id,
        chatbot_version_id=chatbot_version_id,
        run_name=run_name,
        session_factory=session_factory,
    )
    return execute_evaluation_run(
        eval_run_id=eval_run_id,
        session_factory=session_factory,
        answer_function=answer_function,
        connector=connector,
    )


def create_evaluation_run_record(
    eval_dataset_id: int,
    chatbot_version_id: int,
    run_name: str,
    session_factory: Callable[[], Session] = SessionLocal,
) -> int:
    """Create a pending evaluation run and return its id."""
    if not isinstance(eval_dataset_id, int) or eval_dataset_id <= 0:
        raise ValueError("eval_dataset_id must be a positive integer")
    if not isinstance(chatbot_version_id, int) or chatbot_version_id <= 0:
        raise ValueError("chatbot_version_id must be a positive integer")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")

    create_database_tables()

    with session_factory() as database_session:
        _load_dataset_cases(database_session, eval_dataset_id)
        _validate_chatbot_version(database_session, chatbot_version_id)
        eval_run = EvalRun(
            eval_dataset_id=eval_dataset_id,
            chatbot_version_id=chatbot_version_id,
            run_name=run_name.strip(),
            status="pending",
        )
        database_session.add(eval_run)
        database_session.commit()
        database_session.refresh(eval_run)
        return eval_run.id


def execute_evaluation_run(
    eval_run_id: int,
    session_factory: Callable[[], Session] = SessionLocal,
    answer_function: Callable[..., dict] | None = None,
    connector: RAGConnector | None = None,
) -> dict[str, Any]:
    """Execute a previously created evaluation run."""
    if not isinstance(eval_run_id, int) or eval_run_id <= 0:
        raise ValueError("eval_run_id must be a positive integer")

    create_database_tables()

    with session_factory() as database_session:
        eval_run = database_session.get(EvalRun, eval_run_id)
        if eval_run is None:
            raise EvaluationRunError("Evaluation run not found")
        cases = _load_dataset_cases(database_session, eval_run.eval_dataset_id)
        chatbot_version_id = eval_run.chatbot_version_id
        connector_settings = _active_connector_settings(database_session)
        eval_run.status = "running"
        database_session.commit()

    rag_connector = connector
    if rag_connector is None and answer_function is not None:
        rag_connector = FunctionRAGConnector(answer_function)
    if rag_connector is None:
        rag_connector = get_rag_connector(**connector_settings)

    try:
        saved_results = []
        for eval_case in cases:
            answer_result = rag_connector.answer(
                eval_case["question"],
                chatbot_version_id=chatbot_version_id,
                session_factory=session_factory,
            )
            evaluated = evaluate_case(eval_case, answer_result)

            with session_factory() as database_session:
                database_session.add(
                    EvalResult(
                        eval_run_id=eval_run_id,
                        eval_case_id=eval_case["id"],
                        generated_answer=evaluated["generated_answer"],
                        retrieved_chunk_ids=json.dumps(
                            evaluated["retrieved_chunk_ids"],
                        ),
                        retrieved_chunk_keys=json.dumps(
                            evaluated["retrieved_chunk_keys"],
                        ),
                        expected_chunk_hit=evaluated["metrics"]["expected_chunk_hit"],
                        citation_presence=evaluated["metrics"]["citation_presence"],
                        refusal_score=evaluated["metrics"]["refusal_score"],
                        numeric_consistency=evaluated["metrics"]["numeric_consistency"],
                        answer_keyword_score=evaluated["metrics"][
                            "answer_keyword_score"
                        ],
                        hallucination_flag=evaluated["metrics"][
                            "hallucination_flag"
                        ],
                        overall_case_score=evaluated["overall_case_score"],
                        passed=evaluated["passed"],
                        failure_type=evaluated["failure_type"],
                        failure_reason=evaluated["failure_reason"],
                    )
                )
                database_session.commit()

            saved_results.append(evaluated)

        overall_score = round(
            sum(result["overall_case_score"] for result in saved_results)
            / len(saved_results),
            4,
        )
        passed_cases = sum(1 for result in saved_results if result["passed"])

        with session_factory() as database_session:
            eval_run = database_session.get(EvalRun, eval_run_id)
            eval_run.overall_score = overall_score
            eval_run.status = "completed"
            eval_run.completed_at = datetime.now(timezone.utc)
            database_session.commit()

        return {
            "eval_run_id": eval_run_id,
            "total_cases": len(saved_results),
            "passed_cases": passed_cases,
            "failed_cases": len(saved_results) - passed_cases,
            "overall_score": overall_score,
            "status": "completed",
        }
    except Exception:
        with session_factory() as database_session:
            eval_run = database_session.get(EvalRun, eval_run_id)
            if eval_run is not None:
                eval_run.status = "failed"
                eval_run.completed_at = datetime.now(timezone.utc)
                database_session.commit()
        raise
