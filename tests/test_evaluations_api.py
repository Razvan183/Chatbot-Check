"""Tests for evaluation API endpoints."""

import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import evaluations
from app.db.database import Base, get_db
from app.db.models import ChatbotVersion, EvalCase, EvalDataset, EvalResult, EvalRun
from app.main import app


@pytest.fixture
def evaluation_client() -> Generator[TestClient, None, None]:
    """Provide an API client backed by an isolated SQLite database."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    with test_session_factory() as session:
        version = ChatbotVersion(id=1, name="baseline_v1", model_name="mock")
        candidate_version = ChatbotVersion(id=2, name="more_context_v2", model_name="mock")
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
        eval_run = EvalRun(
            id=1,
            eval_dataset_id=1,
            chatbot_version_id=1,
            run_name="Smoke eval",
            status="completed",
            overall_score=0.9,
        )
        candidate_run = EvalRun(
            id=2,
            eval_dataset_id=1,
            chatbot_version_id=2,
            run_name="Candidate eval",
            status="completed",
            overall_score=0.8,
        )
        session.add_all([version, candidate_version])
        session.add(dataset)
        session.add_all([eval_run, candidate_run])
        session.flush()
        session.add_all(
            [
                EvalResult(
                    eval_run_id=1,
                    eval_case_id=1,
                    generated_answer="Employees receive 21 vacation days. [10]",
                    retrieved_chunk_ids=json.dumps([10]),
                    retrieved_chunk_keys=json.dumps(["vacation_policy.md::0"]),
                    expected_chunk_hit=1.0,
                    citation_presence=1.0,
                    refusal_score=None,
                    numeric_consistency=1.0,
                    answer_keyword_score=1.0,
                    hallucination_flag=False,
                    overall_case_score=1.0,
                    passed=True,
                    failure_type="passed",
                    failure_reason="The answer met the evaluation threshold.",
                ),
                EvalResult(
                    eval_run_id=1,
                    eval_case_id=2,
                    generated_answer="The company reimburses gym memberships.",
                    retrieved_chunk_ids=json.dumps([]),
                    retrieved_chunk_keys=json.dumps([]),
                    expected_chunk_hit=None,
                    citation_presence=0.0,
                    refusal_score=0.0,
                    numeric_consistency=1.0,
                    answer_keyword_score=1.0,
                    hallucination_flag=True,
                    overall_case_score=0.0,
                    passed=False,
                    failure_type="answered_unanswerable_question",
                    failure_reason="The answer should have refused.",
                ),
                EvalResult(
                    eval_run_id=2,
                    eval_case_id=1,
                    generated_answer="Employees receive 21 vacation days.",
                    retrieved_chunk_ids=json.dumps([10]),
                    retrieved_chunk_keys=json.dumps(["vacation_policy.md::0"]),
                    expected_chunk_hit=1.0,
                    citation_presence=0.0,
                    refusal_score=None,
                    numeric_consistency=1.0,
                    answer_keyword_score=1.0,
                    hallucination_flag=False,
                    overall_case_score=0.8333,
                    passed=True,
                    failure_type="passed",
                    failure_reason="The answer met the evaluation threshold.",
                ),
                EvalResult(
                    eval_run_id=2,
                    eval_case_id=2,
                    generated_answer="I could not find this information in the provided documents.",
                    retrieved_chunk_ids=json.dumps([]),
                    retrieved_chunk_keys=json.dumps([]),
                    expected_chunk_hit=None,
                    citation_presence=0.0,
                    refusal_score=1.0,
                    numeric_consistency=1.0,
                    answer_keyword_score=1.0,
                    hallucination_flag=False,
                    overall_case_score=1.0,
                    passed=True,
                    failure_type="passed",
                    failure_reason="The answer met the evaluation threshold.",
                ),
            ]
        )
        session.commit()

    def override_get_db() -> Generator[Session, None, None]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()


def test_list_evaluation_datasets(evaluation_client: TestClient) -> None:
    response = evaluation_client.get("/evaluations/datasets")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Tiny Eval Set"
    assert response.json()[0]["case_count"] == 2


def test_list_evaluation_runs_includes_counts(
    evaluation_client: TestClient,
) -> None:
    response = evaluation_client.get("/evaluations/runs")

    assert response.status_code == 200
    payload = response.json()
    smoke_eval = next(run for run in payload if run["run_name"] == "Smoke eval")
    assert smoke_eval["eval_dataset_name"] == "Tiny Eval Set"
    assert smoke_eval["chatbot_version_name"] == "baseline_v1"
    assert smoke_eval["total_cases"] == 2
    assert smoke_eval["passed_cases"] == 1
    assert smoke_eval["failed_cases"] == 1


def test_get_evaluation_run(evaluation_client: TestClient) -> None:
    response = evaluation_client.get("/evaluations/runs/1")

    assert response.status_code == 200
    assert response.json()["overall_score"] == 0.9


def test_list_evaluation_run_results(evaluation_client: TestClient) -> None:
    response = evaluation_client.get("/evaluations/runs/1/results")

    assert response.status_code == 200
    payload = response.json()
    assert [result["eval_case_id"] for result in payload] == [1, 2]
    assert payload[0]["question"] == "How many vacation days after two years?"
    assert payload[0]["retrieved_chunk_ids"] == [10]
    assert payload[0]["retrieved_chunk_keys"] == ["vacation_policy.md::0"]
    assert payload[1]["failure_type"] == "answered_unanswerable_question"


def test_compare_evaluation_runs_returns_regression_summary(
    evaluation_client: TestClient,
) -> None:
    response = evaluation_client.get("/evaluations/runs/1/compare/2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["baseline_run"]["run_name"] == "Smoke eval"
    assert payload["candidate_run"]["run_name"] == "Candidate eval"
    assert payload["overall_score_delta"] == -0.1
    assert payload["passed_cases_delta"] == 1
    assert payload["failed_cases_delta"] == -1
    assert payload["fixed_cases"] == 1
    assert payload["new_failures"] == 0
    assert payload["regressed_cases"] == 1
    assert payload["failure_breakdown"] == [
        {
            "failure_type": "answered_unanswerable_question",
            "baseline_count": 1,
            "candidate_count": 0,
            "delta": -1,
        }
    ]
    assert [case["status"] for case in payload["case_comparisons"]] == [
        "regressed",
        "fixed",
    ]


def test_get_evaluation_scorecard_returns_metrics_and_recommendations(
    evaluation_client: TestClient,
) -> None:
    response = evaluation_client.get("/evaluations/runs/1/scorecard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["run_name"] == "Smoke eval"
    assert {
        metric["name"]: metric["score"]
        for metric in payload["metric_scores"]
    } == {
        "expected_chunk_hit": 1.0,
        "citation_presence": 0.5,
        "refusal_score": 0.0,
        "numeric_consistency": 1.0,
        "answer_keyword_score": 1.0,
    }
    assert payload["failure_summary"] == [
        {"failure_type": "answered_unanswerable_question", "count": 1}
    ]
    recommendation_parameters = {
        recommendation["parameter"]
        for recommendation in payload["recommendations"]
    }
    assert "prompt_template" in recommendation_parameters
    assert "retrieval_score_threshold" in recommendation_parameters


def test_evaluation_comparison_report_returns_html(
    evaluation_client: TestClient,
) -> None:
    response = evaluation_client.get("/evaluations/runs/1/compare/2/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "RAG Evaluation Comparison Report" in response.text
    assert "baseline_v1 vs more_context_v2" in response.text
    assert "Score delta" in response.text
    assert "answered_unanswerable_question" in response.text
    assert "How many vacation days after two years?" in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/evaluations/runs/999",
        "/evaluations/runs/999/results",
        "/evaluations/runs/999/compare/1",
    ],
)
def test_missing_evaluation_run_returns_404(
    evaluation_client: TestClient,
    path: str,
) -> None:
    response = evaluation_client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": "Evaluation run not found"}


def test_create_evaluation_run_uses_runner(
    evaluation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    executed = []

    def fake_create_evaluation_run_record(
        eval_dataset_id: int,
        chatbot_version_id: int,
        run_name: str,
    ) -> int:
        captured["eval_dataset_id"] = eval_dataset_id
        captured["chatbot_version_id"] = chatbot_version_id
        captured["run_name"] = run_name
        return 2

    def fake_execute_evaluation_run(eval_run_id: int) -> dict:
        executed.append(eval_run_id)
        return {"eval_run_id": eval_run_id, "status": "completed"}

    monkeypatch.setattr(
        evaluations,
        "create_evaluation_run_record",
        fake_create_evaluation_run_record,
    )
    monkeypatch.setattr(evaluations, "execute_evaluation_run", fake_execute_evaluation_run)

    response = evaluation_client.post(
        "/evaluations/runs",
        json={
            "eval_dataset_id": 1,
            "chatbot_version_id": 1,
            "run_name": "API eval",
        },
    )

    assert response.status_code == 202
    assert response.json()["eval_run_id"] == 2
    assert response.json()["status"] == "pending"
    assert executed == [2]
    assert captured == {
        "eval_dataset_id": 1,
        "chatbot_version_id": 1,
        "run_name": "API eval",
    }
