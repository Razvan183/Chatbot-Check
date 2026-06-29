"""Tests for dataset authoring LLM provider configuration."""

import httpx
import pytest

from app.evaluation.dataset_generator import (
    DatasetGeneratorError,
    DisabledDatasetGenerator,
    GeminiDatasetGenerator,
    MockDatasetGenerator,
    extract_response_text,
    get_dataset_generator_provider,
)


def test_default_dataset_generator_is_disabled() -> None:
    provider = get_dataset_generator_provider("disabled")

    assert isinstance(provider, DisabledDatasetGenerator)
    with pytest.raises(DatasetGeneratorError):
        provider.generate_text("Generate cases")


def test_mock_dataset_generator_returns_configured_text() -> None:
    provider = MockDatasetGenerator(response_text='[{"id":"q001"}]')

    assert provider.generate_text("Generate cases") == '[{"id":"q001"}]'


def test_gemini_dataset_generator_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.evaluation.dataset_generator.DATASET_GENERATOR_API_KEY",
        "test-key",
    )

    assert isinstance(get_dataset_generator_provider("gemini"), GeminiDatasetGenerator)


def test_gemini_dataset_generator_requires_api_key() -> None:
    with pytest.raises(DatasetGeneratorError):
        GeminiDatasetGenerator(api_key="")


def test_gemini_dataset_generator_calls_generate_content_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_post(
        url: str,
        headers: dict,
        json: dict,
        timeout: float,
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"output_text": '[{"question":"What is the policy?"}]'},
        )

    monkeypatch.setattr("app.evaluation.dataset_generator.httpx.post", fake_post)
    provider = GeminiDatasetGenerator(
        api_key="test-key",
        model="gemini-test",
        timeout_seconds=12,
        max_output_tokens=500,
    )

    assert provider.generate_text("Generate evaluation cases") == (
        '[{"question":"What is the policy?"}]'
    )
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-test:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["json"] == {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Generate evaluation cases",
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 500,
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }
    assert captured["timeout"] == 12


def test_gemini_dataset_generator_accepts_prefixed_model_name() -> None:
    provider = GeminiDatasetGenerator(
        api_key="test-key",
        model="models/gemini-test",
    )

    assert provider.endpoint_url == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-test:generateContent"
    )


def test_extract_response_text_from_output_content_items() -> None:
    payload = {
        "output": [
            {
                "content": [
                    {"text": "First part"},
                    {"text": "Second part"},
                ]
            }
        ]
    }

    assert extract_response_text(payload) == "First part\nSecond part"


def test_extract_response_text_from_gemini_candidates() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "First part"},
                        {"text": "Second part"},
                    ]
                }
            }
        ]
    }

    assert extract_response_text(payload) == "First part\nSecond part"
