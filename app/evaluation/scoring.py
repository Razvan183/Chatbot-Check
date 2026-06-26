"""Combine evaluation metrics into case-level scores."""


def has_hallucination_flag(
    metrics: dict,
    should_be_answerable: bool,
    question_type: str,
) -> bool:
    """Return a simple hallucination-risk flag from metric values."""
    if not should_be_answerable and metrics.get("refusal_score") == 0.0:
        return True
    if metrics.get("numeric_consistency") is not None and (
        metrics["numeric_consistency"] < 1.0
    ):
        return True
    if question_type == "citation_required" and metrics.get("citation_presence") == 0.0:
        return True
    return False


def _metric(metrics: dict, name: str, default: float = 1.0) -> float:
    """Read a metric value, treating None as a neutral default."""
    value = metrics.get(name)
    return default if value is None else float(value)


def calculate_case_score(
    metrics: dict,
    should_be_answerable: bool,
    question_type: str = "factual",
) -> float:
    """Calculate a weighted score between 0.0 and 1.0 for one eval case."""
    hallucination_flag = bool(
        metrics.get(
            "hallucination_flag",
            has_hallucination_flag(metrics, should_be_answerable, question_type),
        )
    )

    if should_be_answerable:
        score = (
            _metric(metrics, "answer_keyword_score") * 0.30
            + _metric(metrics, "numeric_consistency") * 0.25
            + _metric(metrics, "citation_presence") * 0.15
            + _metric(metrics, "expected_chunk_hit") * 0.20
        )
        score = score / 0.90
        if hallucination_flag:
            score -= 0.20
    else:
        refusal_score = _metric(metrics, "refusal_score", default=0.0)
        score = refusal_score * 0.70
        score += 0.30 if not hallucination_flag else 0.0

    return round(max(0.0, min(1.0, score)), 4)
