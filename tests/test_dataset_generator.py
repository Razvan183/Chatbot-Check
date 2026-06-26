"""Tests for dataset authoring LLM provider configuration."""

import httpx
import pytest

from app.evaluation.dataset_generator import (
    DatasetGeneratorError,
    DisabledDatasetGenerator,
    MockDatasetGenerator,
    OpenAIDatasetGenerator,
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


def test_openai_dataset_generator_requires_api_key() -> None:
    with pytest.raises(DatasetGeneratorError):
        OpenAIDatasetGenerator(api_key="")


def test_openai_dataset_generator_calls_responses_api(
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
    provider = OpenAIDatasetGenerator(
        api_key="test-key",
        model="gpt-test",
        timeout_seconds=12,
        max_output_tokens=500,
    )

    assert provider.generate_text("Generate evaluation cases") == (
        '[{"question":"What is the policy?"}]'
    )
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"] == {
        "model": "gpt-test",
        "input": "Generate evaluation cases",
        "max_output_tokens": 500,
    }
    assert captured["timeout"] == 12


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
