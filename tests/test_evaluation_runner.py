"""Tests for running evaluation datasets."""

import json
from collections.abc import Callable

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.connectors.base import RAGAnswer
from app.db.database import Base
from app.db.models import ChatbotVersion, EvalCase, EvalDataset, EvalResult, EvalRun
from app.db.models import RAGConnectorConfig
from app.evaluation.runner import (
    EvaluationDatasetNotFoundError,
    evaluate_case,
    run_evaluation,
)


@pytest.fixture
def evaluation_session_factory() -> Callable[[], Session]:
    """Create an isolated database with one dataset and version."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        session.add(ChatbotVersion(id=1, name="baseline_v1", model_name="mock"))
        dataset = EvalDataset(
            id=1,
            name="Tiny Eval Set",
            description="Small fixture",
            domain="company_policy",
        )
        dataset.cases = [
            EvalCase(
                id=1,
                question="How many vacation days after two years?",
                expected_answer="Employees receive 21 vacation days.",
                expected_chunk_ids=json.dumps([10]),
                expected_chunk_keys=json.dumps(["vacation_policy.md::0"]),
                question_type="factual",
                difficulty="easy",
                should_be_answerable=True,
            ),
            EvalCase(
                id=2,
                question="Does the company reimburse gym memberships?",
                expected_answer=None,
                expected_chunk_ids=json.dumps([]),
                expected_chunk_keys=json.dumps([]),
                question_type="unanswerable",
                difficulty="easy",
                should_be_answerable=False,
            ),
        ]
        session.add(dataset)
        session.commit()

    try:
        yield test_session_factory
    finally:
        test_engine.dispose()


def test_evaluate_case_returns_metrics_score_and_failure_type() -> None:
    eval_case = {
        "id": 1,
        "question": "How many vacation days after two years?",
        "expected_answer": "Employees receive 21 vacation days.",
        "expected_chunk_ids": [10],
        "expected_chunk_keys": ["vacation_policy.md::0"],
        "question_type": "factual",
        "difficulty": "easy",
        "should_be_answerable": True,
    }
    answer_result = {
        "answer": "Employees receive 21 vacation days. [10]",
        "retrieved_chunks": [
            {
                "chunk_id": 10,
                "chunk_key": "vacation_policy.md::0",
                "document_id": 1,
                "filename": "vacation_policy.md",
                "chunk_text": "Employees receive 21 vacation days.",
                "score": 0.9,
            }
        ],
    }

    result = evaluate_case(eval_case, answer_result)

    assert result["overall_case_score"] >= 0.75
    assert result["passed"] is True
    assert result["failure_type"] == "passed"
    assert result["metrics"]["expected_chunk_hit"] == 1.0


def test_run_evaluation_persists_run_and_results(
    evaluation_session_factory: Callable[[], Session],
) -> None:
    def fake_answer_question(
        question: str,
        chatbot_version_id: int,
        session_factory: Callable[[], Session],
    ) -> dict:
        if "gym" in question:
            return {
                "question": question,
                "answer": "I could not find this information in the provided documents.",
                "retrieved_chunks": [],
                "citations": [],
                "latency_ms": 1,
            }

        return {
            "question": question,
            "answer": "Employees receive 21 vacation days. [10]",
            "retrieved_chunks": [
                {
                    "chunk_id": 10,
                    "chunk_key": "vacation_policy.md::0",
                    "document_id": 1,
                    "filename": "vacation_policy.md",
                    "chunk_text": "Employees receive 21 vacation days.",
                    "score": 0.9,
                }
            ],
            "citations": [10],
            "latency_ms": 1,
        }

    summary = run_evaluation(
        eval_dataset_id=1,
        chatbot_version_id=1,
        run_name="Smoke eval",
        session_factory=evaluation_session_factory,
        answer_function=fake_answer_question,
    )

    with evaluation_session_factory() as session:
        eval_run = session.get(EvalRun, summary["eval_run_id"])
        result_count = session.scalar(select(func.count()).select_from(EvalResult))
        results = list(session.scalars(select(EvalResult).order_by(EvalResult.id)))

    assert summary == {
        "eval_run_id": eval_run.id,
        "total_cases": 2,
        "passed_cases": 2,
        "failed_cases": 0,
        "overall_score": 1.0,
        "status": "completed",
    }
    assert eval_run.status == "completed"
    assert result_count == 2
    assert [result.failure_type for result in results] == ["passed", "passed"]


def test_run_evaluation_can_use_connector_object(
    evaluation_session_factory: Callable[[], Session],
) -> None:
    class FakeConnector:
        def answer(
            self,
            question: str,
            chatbot_version_id: int,
            session_factory: Callable[[], Session],
        ) -> RAGAnswer:
            del chatbot_version_id, session_factory
            if "gym" in question:
                return {
                    "question": question,
                    "answer": "I could not find this information in the provided documents.",
                    "retrieved_chunks": [],
                    "citations": [],
                    "latency_ms": 1,
                }

            return {
                "question": question,
                "answer": "Employees receive 21 vacation days. [10]",
                "retrieved_chunks": [
                    {
                        "chunk_id": 10,
                        "chunk_key": "vacation_policy.md::0",
                        "document_id": 1,
                        "filename": "vacation_policy.md",
                        "chunk_text": "Employees receive 21 vacation days.",
                        "score": 0.9,
                    }
                ],
                "citations": [10],
                "latency_ms": 1,
            }

    summary = run_evaluation(
        eval_dataset_id=1,
        chatbot_version_id=1,
        run_name="Connector eval",
        session_factory=evaluation_session_factory,
        connector=FakeConnector(),
    )

    assert summary["status"] == "completed"
    assert summary["passed_cases"] == 2


def test_run_evaluation_uses_active_persisted_connector_settings(
    evaluation_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeConnector:
        def answer(
            self,
            question: str,
            chatbot_version_id: int,
            session_factory: Callable[[], Session],
        ) -> RAGAnswer:
            del chatbot_version_id, session_factory
            if "gym" in question:
                return {
                    "question": question,
                    "answer": "I could not find this information in the provided documents.",
                    "retrieved_chunks": [],
                    "citations": [],
                    "latency_ms": 1,
                }
            return {
                "question": question,
                "answer": "Employees receive 21 vacation days. [10]",
                "retrieved_chunks": [
                    {
                        "chunk_id": 10,
                        "chunk_key": "vacation_policy.md::0",
                        "document_id": 1,
                        "filename": "vacation_policy.md",
                        "chunk_text": "Employees receive 21 vacation days.",
                        "score": 0.9,
                    }
                ],
                "citations": [10],
                "latency_ms": 1,
            }

    def fake_get_rag_connector(**kwargs):
        captured.update(kwargs)
        return FakeConnector()

    monkeypatch.setattr(
        "app.evaluation.runner.get_rag_connector",
        fake_get_rag_connector,
    )
    with evaluation_session_factory() as session:
        session.add(
            RAGConnectorConfig(
                connector_type="http",
                http_url="http://rag.local/chat",
                timeout_seconds=9,
                active=True,
            )
        )
        session.commit()

    summary = run_evaluation(
        eval_dataset_id=1,
        chatbot_version_id=1,
        run_name="Persisted connector eval",
        session_factory=evaluation_session_factory,
    )

    assert summary["status"] == "completed"
    assert captured == {
        "connector_name": "http",
        "http_url": "http://rag.local/chat",
        "timeout_seconds": 9,
    }


def test_run_evaluation_rejects_missing_dataset(
    evaluation_session_factory: Callable[[], Session],
) -> None:
    with pytest.raises(EvaluationDatasetNotFoundError):
        run_evaluation(
            eval_dataset_id=999,
            chatbot_version_id=1,
            run_name="Missing dataset",
            session_factory=evaluation_session_factory,
        )
