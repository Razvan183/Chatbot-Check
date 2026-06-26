"""Answer generation for the RAG chatbot."""

import re
from typing import Any

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, USE_MOCK_LLM
from app.rag.prompts import REFUSAL_MESSAGE


WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")


def _validate_prompt(prompt: str) -> None:
    """Validate prompt text before sending it to a generator."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")


def _important_words(text: str) -> set[str]:
    """Return simple lowercase tokens used for mock relevance checks."""
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "do",
        "does",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "of",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "with",
    }
    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in stopwords
    }


def _split_sentences(text: str) -> list[str]:
    """Split text into short candidate answer sentences."""
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.findall(text)
        if sentence.strip()
    ]


def _best_sentence(
    question: str,
    chunk_text: str,
    minimum_overlap: int = 1,
) -> str | None:
    """Choose the sentence with the strongest simple word overlap."""
    question_words = _important_words(question)
    if not question_words:
        return None

    best_sentence = None
    best_overlap = 0
    for sentence in _split_sentences(chunk_text):
        sentence_words = _important_words(sentence)
        overlap = len(question_words & sentence_words)
        if overlap > best_overlap:
            best_sentence = sentence
            best_overlap = overlap

    return best_sentence if best_overlap >= minimum_overlap else None


def _parse_prompt_for_mock(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract question and context chunks from prompts built by prompts.py."""
    question_match = re.search(r"\nQuestion:\n(?P<question>.*?)\n\nAnswer:\s*$", prompt, re.DOTALL)
    context_match = re.search(r"\nContext:\n(?P<context>.*?)\n\nQuestion:\n", prompt, re.DOTALL)

    question = question_match.group("question").strip() if question_match else prompt.strip()
    context = context_match.group("context").strip() if context_match else ""

    if not context or context == "No relevant context was retrieved.":
        return question, []

    chunks: list[dict[str, Any]] = []
    chunk_matches = re.finditer(
        r"(?:^|\n\n)\[(?P<chunk_id>\d+)\]\s+(?P<filename>[^\n]+)\n"
        r"(?P<chunk_text>.*?)(?=\n\n\[\d+\]\s+|\Z)",
        context,
        re.DOTALL,
    )
    for match in chunk_matches:
        chunks.append(
            {
                "chunk_id": int(match.group("chunk_id")),
                "filename": match.group("filename").strip(),
                "chunk_text": match.group("chunk_text").strip(),
            }
        )

    return question, chunks


def _mock_minimum_overlap(temperature: float | None) -> int:
    """Use temperature as a deterministic strictness setting for mock answers."""
    if temperature is None:
        return 1
    if not isinstance(temperature, int | float):
        raise ValueError("temperature must be numeric")
    if temperature <= 0:
        return 2
    return 1


def generate_mock_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    temperature: float | None = None,
) -> str:
    """Generate a deterministic answer from the best available retrieved chunk."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(retrieved_chunks, list):
        raise ValueError("retrieved_chunks must be a list")

    minimum_overlap = _mock_minimum_overlap(temperature)

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            raise ValueError("retrieved_chunks must contain dictionaries")

        chunk_text = str(chunk.get("chunk_text", "")).strip()
        chunk_id = chunk.get("chunk_id")
        if not chunk_text or chunk_id is None:
            continue

        sentence = _best_sentence(
            question,
            chunk_text,
            minimum_overlap=minimum_overlap,
        )
        if sentence is not None:
            return f"{sentence} [{chunk_id}]"

    return REFUSAL_MESSAGE


def generate_with_ollama(
    prompt: str,
    model_name: str = OLLAMA_MODEL,
    temperature: float | None = None,
) -> str:
    """Generate an answer with a local Ollama model."""
    _validate_prompt(prompt)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if temperature is not None and not isinstance(temperature, int | float):
        raise ValueError("temperature must be numeric")

    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if temperature is not None:
        payload["options"] = {"temperature": float(temperature)}

    response = httpx.post(
        endpoint,
        json=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()

    answer = str(data.get("response", "")).strip()
    if not answer:
        return "Local LLM generation failed: Ollama returned an empty response."

    return answer


def generate_answer(
    prompt: str,
    generation_mode: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    question: str | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Generate an answer from a prompt using mock mode or local Ollama."""
    _validate_prompt(prompt)
    if generation_mode is not None and not isinstance(generation_mode, str):
        raise ValueError("generation_mode must be a string")

    if generation_mode is None:
        generation_mode = "mock" if USE_MOCK_LLM else "ollama"

    normalized_mode = generation_mode.strip().lower()
    if normalized_mode == "mock":
        if question is None or retrieved_chunks is None:
            parsed_question, parsed_chunks = _parse_prompt_for_mock(prompt)
            question = parsed_question
            retrieved_chunks = parsed_chunks
        return generate_mock_answer(question, retrieved_chunks, temperature=temperature)

    if normalized_mode != "ollama":
        raise ValueError("generation_mode must be 'mock' or 'ollama'")

    try:
        return generate_with_ollama(
            prompt,
            model_name=model_name or OLLAMA_MODEL,
            temperature=temperature,
        )
    except httpx.HTTPError as exc:
        return f"Local LLM generation failed: {exc}"
