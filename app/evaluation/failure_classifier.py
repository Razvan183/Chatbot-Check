"""Classify evaluation failures into readable categories."""


def classify_failure(case: dict, metrics: dict, score: float) -> tuple[str, str]:
    """Return a failure type and short explanation for one evaluated answer."""
    if score >= 0.75:
        return "passed", "The answer met the evaluation threshold."

    if not case["should_be_answerable"] and metrics.get("refusal_score") == 0.0:
        return (
            "answered_unanswerable_question",
            "The answer should have refused because the documents do not contain this information.",
        )

    if (
        case.get("question_type") == "citation_required"
        and metrics.get("citation_presence") == 0.0
    ):
        return "missing_citation", "The answer did not include a required chunk citation."

    if metrics.get("numeric_consistency") is not None and (
        metrics["numeric_consistency"] < 1.0
    ):
        return (
            "wrong_number",
            "The generated answer contains numbers that are not supported by the retrieved context.",
        )

    if metrics.get("expected_chunk_hit") == 0.0:
        return (
            "retrieval_failure",
            "The retriever did not return the expected supporting chunks.",
        )

    if metrics.get("answer_keyword_score") is not None and (
        metrics["answer_keyword_score"] < 0.4
    ):
        return (
            "low_answer_overlap",
            "The generated answer has low keyword overlap with the expected answer.",
        )

    if metrics.get("hallucination_flag"):
        return (
            "possible_hallucination",
            "The answer triggered a simple hallucination-risk rule.",
        )

    return "unknown_failure", "The answer failed the threshold for an unspecified reason."
