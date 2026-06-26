"""Tests for mock and local-LLM answer generation."""

import httpx
import pytest

from app.rag import generator
from app.rag.generator import generate_answer, generate_mock_answer, generate_with_ollama
from app.rag.prompts import REFUSAL_MESSAGE, build_rag_prompt


def test_generate_mock_answer_returns_matching_sentence_with_citation() -> None:
    answer = generate_mock_answer(
        "How many vacation days do employees receive after two years?",
        [
            {
                "chunk_id": 4,
                "filename": "vacation_policy.md",
                "chunk_text": (
                    "Vacation requests require 10 business days notice. "
                    "Employees receive 21 vacation days after two years."
                ),
                "score": 0.92,
            }
        ],
    )

    assert answer == "Employees receive 21 vacation days after two years. [4]"


def test_generate_mock_answer_refuses_without_matching_context() -> None:
    answer = generate_mock_answer(
        "Does the company reimburse gym memberships?",
        [
            {
                "chunk_id": 8,
                "filename": "equipment_policy.md",
                "chunk_text": "Lost equipment must be reported within 24 hours.",
            }
        ],
    )

    assert answer == REFUSAL_MESSAGE


def test_generate_mock_answer_uses_temperature_as_strictness() -> None:
    chunks = [
        {
            "chunk_id": 8,
            "filename": "equipment_policy.md",
            "chunk_text": "Lost equipment must be reported within 24 hours.",
        }
    ]

    assert (
        generate_mock_answer(
            "When should laptops be reported?",
            chunks,
            temperature=0.2,
        )
        == "Lost equipment must be reported within 24 hours. [8]"
    )
    assert (
        generate_mock_answer(
            "When should laptops be reported?",
            chunks,
            temperature=0.0,
        )
        == REFUSAL_MESSAGE
    )


def test_generate_answer_uses_mock_mode_from_built_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "USE_MOCK_LLM", True)
    prompt = build_rag_prompt(
        "When should lost equipment be reported?",
        [
            {
                "chunk_id": 9,
                "filename": "equipment_policy.md",
                "chunk_text": "Lost equipment must be reported within 24 hours.",
            }
        ],
    )

    assert generate_answer(prompt) == "Lost equipment must be reported within 24 hours. [9]"


def test_generate_answer_calls_ollama_when_mock_mode_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator, "USE_MOCK_LLM", False)
    monkeypatch.setattr(
        generator,
        "generate_with_ollama",
        lambda prompt, model_name=generator.OLLAMA_MODEL, temperature=None: "Ollama answer",
    )

    assert generate_answer("Prompt text") == "Ollama answer"


def test_generate_answer_returns_clear_message_when_ollama_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator, "USE_MOCK_LLM", False)

    def fail_ollama(
        _: str,
        model_name: str = generator.OLLAMA_MODEL,
        temperature: float | None = None,
    ) -> str:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(generator, "generate_with_ollama", fail_ollama)

    assert generate_answer("Prompt text").startswith("Local LLM generation failed:")


def test_generate_with_ollama_posts_to_local_generate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "Generated answer [1]"}

    def fake_post(url, json, timeout):
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(generator.httpx, "post", fake_post)

    answer = generate_with_ollama("Prompt text", model_name="llama3.2:3b")

    assert answer == "Generated answer [1]"
    assert captured_request == {
        "url": "http://localhost:11434/api/generate",
        "json": {
            "model": "llama3.2:3b",
            "prompt": "Prompt text",
            "stream": False,
        },
        "timeout": 60.0,
    }


def test_generate_with_ollama_sends_temperature_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "Generated answer [1]"}

    def fake_post(url, json, timeout):
        captured_request["json"] = json
        return FakeResponse()

    monkeypatch.setattr(generator.httpx, "post", fake_post)

    assert (
        generate_with_ollama(
            "Prompt text",
            model_name="qwen3:8b",
            temperature=0.1,
        )
        == "Generated answer [1]"
    )
    assert captured_request["json"]["options"] == {"temperature": 0.1}


def test_generate_answer_can_use_mock_mode_without_parsing_prompt() -> None:
    answer = generate_answer(
        "Custom prompt without the default parser shape",
        generation_mode="mock",
        question="When should lost equipment be reported?",
        retrieved_chunks=[
            {
                "chunk_id": 9,
                "filename": "equipment_policy.md",
                "chunk_text": "Lost equipment must be reported within 24 hours.",
            }
        ],
    )

    assert answer == "Lost equipment must be reported within 24 hours. [9]"


@pytest.mark.parametrize("prompt", ["", "   ", 123])
def test_generate_answer_rejects_invalid_prompt(prompt) -> None:
    with pytest.raises(ValueError, match="prompt"):
        generate_answer(prompt)
