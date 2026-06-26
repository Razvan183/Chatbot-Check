"""Tests for RAG prompt construction."""

import pytest

from app.rag.prompts import build_rag_prompt


def test_build_rag_prompt_includes_rules_context_and_question() -> None:
    retrieved_chunks = [
        {
            "chunk_id": 7,
            "filename": "vacation_policy.md",
            "chunk_text": "Employees receive 21 vacation days after two years.",
            "score": 0.91,
        },
        {
            "chunk_id": 12,
            "filename": "equipment_policy.md",
            "chunk_text": "Lost equipment must be reported within 24 hours.",
            "score": 0.84,
        },
    ]

    prompt = build_rag_prompt(
        "How many vacation days do employees receive after two years?",
        retrieved_chunks,
    )

    assert "You are a company policy assistant." in prompt
    assert "using only the provided context" in prompt
    assert "I could not find this information in the provided documents." in prompt
    assert "Do not invent policies, numbers, benefits, dates, or procedures." in prompt
    assert "Include citations using the format [chunk_id]." in prompt
    assert "[7] vacation_policy.md" in prompt
    assert "[12] equipment_policy.md" in prompt
    assert "21 vacation days" in prompt
    assert "reported within 24 hours" in prompt
    assert "Question:\nHow many vacation days" in prompt
    assert prompt.endswith("Answer:")


def test_build_rag_prompt_handles_empty_context() -> None:
    prompt = build_rag_prompt("Does the company offer free lunch?", [])

    assert "Context:\nNo relevant context was retrieved." in prompt
    assert "Question:\nDoes the company offer free lunch?" in prompt


def test_build_rag_prompt_supports_custom_template() -> None:
    prompt = build_rag_prompt(
        "Does the company offer free lunch?",
        [],
        prompt_template=(
            "Evidence:\n{context}\n\nAsk: {question}\n\n"
            "Refuse with: {refusal_message}"
        ),
    )

    assert "Evidence:\nNo relevant context was retrieved." in prompt
    assert "Ask: Does the company offer free lunch?" in prompt
    assert "I could not find this information" in prompt


@pytest.mark.parametrize("question", ["", "   ", 123])
def test_build_rag_prompt_rejects_invalid_question(question) -> None:
    with pytest.raises(ValueError, match="question"):
        build_rag_prompt(question, [])


@pytest.mark.parametrize(
    "retrieved_chunks",
    [
        "not a list",
        [{"chunk_id": 1, "chunk_text": "Missing filename"}],
        [{"chunk_id": 1, "filename": "policy.md", "chunk_text": ""}],
    ],
)
def test_build_rag_prompt_rejects_invalid_chunks(retrieved_chunks) -> None:
    with pytest.raises(ValueError, match="retrieved_chunks"):
        build_rag_prompt("What is the policy?", retrieved_chunks)


def test_build_rag_prompt_rejects_unknown_template_placeholder() -> None:
    with pytest.raises(ValueError, match="Unknown prompt template placeholder"):
        build_rag_prompt(
            "What is the policy?",
            [],
            prompt_template="Question: {question}\nBad: {unknown}",
        )
