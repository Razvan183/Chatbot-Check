"""Tests for combining evaluation metrics into scores."""

from app.evaluation.scoring import calculate_case_score, has_hallucination_flag


def test_calculate_case_score_for_answerable_case() -> None:
    score = calculate_case_score(
        {
            "answer_keyword_score": 1.0,
            "numeric_consistency": 1.0,
            "citation_presence": 1.0,
            "expected_chunk_hit": 0.5,
            "hallucination_flag": False,
        },
        should_be_answerable=True,
    )

    assert score == 0.8889


def test_calculate_case_score_penalizes_hallucination_flag() -> None:
    score = calculate_case_score(
        {
            "answer_keyword_score": 1.0,
            "numeric_consistency": 0.5,
            "citation_presence": 1.0,
            "expected_chunk_hit": 1.0,
            "hallucination_flag": True,
        },
        should_be_answerable=True,
    )

    assert score == 0.6611


def test_calculate_case_score_for_unanswerable_case() -> None:
    assert (
        calculate_case_score(
            {"refusal_score": 1.0, "hallucination_flag": False},
            should_be_answerable=False,
        )
        == 1.0
    )
    assert (
        calculate_case_score(
            {"refusal_score": 0.0, "hallucination_flag": True},
            should_be_answerable=False,
        )
        == 0.0
    )


def test_has_hallucination_flag_detects_simple_risks() -> None:
    assert has_hallucination_flag(
        {"refusal_score": 0.0},
        should_be_answerable=False,
        question_type="unanswerable",
    )
    assert has_hallucination_flag(
        {"numeric_consistency": 0.5},
        should_be_answerable=True,
        question_type="factual",
    )
    assert has_hallucination_flag(
        {"citation_presence": 0.0},
        should_be_answerable=True,
        question_type="citation_required",
    )
