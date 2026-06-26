"""Tests for custom evaluation metrics."""

import pytest

from app.evaluation.metrics import (
    answer_keyword_score,
    citation_presence,
    expected_chunk_hit,
    extract_numbers,
    numeric_consistency,
    refusal_correctness,
)


def test_citation_presence_detects_numeric_citations() -> None:
    assert citation_presence("Employees receive 21 days. [12]") == 1.0
    assert citation_presence("Employees receive 21 days.") == 0.0


def test_refusal_correctness_only_scores_unanswerable_cases() -> None:
    assert refusal_correctness("I could not find this information.", False) == 1.0
    assert refusal_correctness("The company reimburses gym memberships.", False) == 0.0
    assert refusal_correctness("Employees receive 21 days. [1]", True) is None


def test_expected_chunk_hit_scores_expected_overlap() -> None:
    assert expected_chunk_hit([], [1, 2]) is None
    assert expected_chunk_hit([1, 2], [2, 3]) == 0.5
    assert expected_chunk_hit([1, 2], [1, 2, 3]) == 1.0


def test_extract_numbers_normalizes_commas() -> None:
    assert extract_numbers("Limit is 150, then 10.5 later.") == {"150", "10.5"}


@pytest.mark.parametrize(
    ("answer", "context", "score"),
    [
        ("Employees receive 21 days.", "Policy says 21 days.", 1.0),
        ("Employees receive 21 days and carry 5.", "Policy says 21 days.", 0.5),
        ("Employees receive 30 days.", "Policy says 21 days.", 0.0),
        ("Employees receive vacation days.", "Policy says 21 days.", 1.0),
    ],
)
def test_numeric_consistency(answer: str, context: str, score: float) -> None:
    assert numeric_consistency(answer, context) == score


def test_answer_keyword_score_uses_expected_answer_overlap() -> None:
    assert answer_keyword_score(None, "Any answer") == 1.0
    assert answer_keyword_score(
        "Employees receive paid vacation days after two years.",
        "Full-time employees receive vacation days after two years. [1]",
    ) > 0.6
    assert answer_keyword_score(
        "Employees receive paid vacation days after two years.",
        "The hotel receipt must be itemized. [2]",
    ) < 0.4
