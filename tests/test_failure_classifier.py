"""Tests for evaluation failure classification."""

from app.evaluation.failure_classifier import classify_failure


def test_classify_failure_returns_passed_for_high_score() -> None:
    assert classify_failure(
        {"should_be_answerable": True, "question_type": "factual"},
        {},
        0.9,
    )[0] == "passed"


def test_classify_failure_prioritizes_answered_unanswerable_question() -> None:
    failure_type, reason = classify_failure(
        {"should_be_answerable": False, "question_type": "unanswerable"},
        {"refusal_score": 0.0},
        0.0,
    )

    assert failure_type == "answered_unanswerable_question"
    assert "should have refused" in reason


def test_classify_failure_detects_missing_citation() -> None:
    assert (
        classify_failure(
            {"should_be_answerable": True, "question_type": "citation_required"},
            {"citation_presence": 0.0},
            0.5,
        )[0]
        == "missing_citation"
    )


def test_classify_failure_detects_wrong_number() -> None:
    assert (
        classify_failure(
            {"should_be_answerable": True, "question_type": "factual"},
            {"numeric_consistency": 0.5},
            0.5,
        )[0]
        == "wrong_number"
    )


def test_classify_failure_detects_retrieval_failure() -> None:
    assert (
        classify_failure(
            {"should_be_answerable": True, "question_type": "factual"},
            {"expected_chunk_hit": 0.0},
            0.5,
        )[0]
        == "retrieval_failure"
    )


def test_classify_failure_detects_low_answer_overlap() -> None:
    assert (
        classify_failure(
            {"should_be_answerable": True, "question_type": "factual"},
            {"answer_keyword_score": 0.2},
            0.5,
        )[0]
        == "low_answer_overlap"
    )
