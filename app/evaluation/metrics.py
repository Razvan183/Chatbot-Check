"""Small deterministic metrics for evaluating RAG answers."""

import re


CITATION_PATTERN = re.compile(r"\[\d+\]")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")

REFUSAL_PHRASES = (
    "could not find",
    "not mentioned",
    "not present",
    "provided documents do not",
    "no information",
    "not available in the documents",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "with",
}


def citation_presence(answer: str) -> float:
    """Return 1.0 when an answer includes at least one numeric citation."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    return 1.0 if CITATION_PATTERN.search(answer) else 0.0


def answer_refuses(answer: str) -> bool:
    """Return whether an answer appears to refuse due to missing evidence."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    normalized = answer.lower()
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def refusal_correctness(answer: str, should_be_answerable: bool) -> float | None:
    """Score refusal behavior for unanswerable cases."""
    if not isinstance(should_be_answerable, bool):
        raise TypeError("should_be_answerable must be a bool")
    if should_be_answerable:
        return None

    return 1.0 if answer_refuses(answer) else 0.0


def expected_chunk_hit(
    expected_chunk_ids: list[int],
    retrieved_chunk_ids: list[int],
) -> float | None:
    """Return the fraction of expected chunks found in retrieval results."""
    if not isinstance(expected_chunk_ids, list) or not isinstance(
        retrieved_chunk_ids,
        list,
    ):
        raise TypeError("chunk id inputs must be lists")
    if not expected_chunk_ids:
        return None

    expected = set(expected_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    return len(expected & retrieved) / len(expected)


def expected_chunk_key_hit(
    expected_chunk_keys: list[str],
    retrieved_chunk_keys: list[str],
) -> float | None:
    """Return the fraction of expected stable chunk keys found in retrieval results."""
    if not isinstance(expected_chunk_keys, list) or not isinstance(
        retrieved_chunk_keys,
        list,
    ):
        raise TypeError("chunk key inputs must be lists")
    if not expected_chunk_keys:
        return None

    expected = {str(key) for key in expected_chunk_keys}
    retrieved = {str(key) for key in retrieved_chunk_keys}
    return len(expected & retrieved) / len(expected)


def extract_numbers(text: str) -> set[str]:
    """Extract simple normalized numeric strings from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return {match.replace(",", ".") for match in NUMBER_PATTERN.findall(text)}


def numeric_consistency(answer: str, context: str) -> float:
    """Score whether numbers in the answer are supported by retrieved context."""
    answer_without_citations = CITATION_PATTERN.sub("", answer)
    answer_numbers = extract_numbers(answer_without_citations)
    if not answer_numbers:
        return 1.0

    context_numbers = extract_numbers(context)
    unsupported_count = len(answer_numbers - context_numbers)
    if unsupported_count == 0:
        return 1.0
    if unsupported_count < len(answer_numbers):
        return 0.5
    return 0.0


def important_words(text: str) -> set[str]:
    """Return lowercased content words used for keyword overlap scoring."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in STOPWORDS
    }


def answer_keyword_score(expected_answer: str | None, generated_answer: str) -> float:
    """Score overlap between expected answer keywords and generated answer."""
    if expected_answer is None or not expected_answer.strip():
        return 1.0

    expected_words = important_words(expected_answer)
    if not expected_words:
        return 1.0

    generated_words = important_words(generated_answer)
    return len(expected_words & generated_words) / len(expected_words)
