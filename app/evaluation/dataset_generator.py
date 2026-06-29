"""Provider layer for the LLM that authors candidate evaluation datasets."""

from typing import Any, Protocol

import httpx

from app.config import (
    DATASET_GENERATOR_API_KEY,
    DATASET_GENERATOR_MAX_OUTPUT_TOKENS,
    DATASET_GENERATOR_MODE,
    DATASET_GENERATOR_MODEL,
    DATASET_GENERATOR_TIMEOUT_SECONDS,
)


GEMINI_GENERATE_CONTENT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class DatasetGeneratorError(RuntimeError):
    """Raised when dataset generation is unavailable or fails."""


class DatasetGeneratorProvider(Protocol):
    """Interface for an LLM used to author candidate eval cases."""

    def generate_text(self, prompt: str) -> str:
        """Generate text from a dataset-authoring prompt."""


class DisabledDatasetGenerator:
    """Provider used when autonomous dataset generation is intentionally off."""

    def generate_text(self, prompt: str) -> str:
        """Raise a clear error because generation is disabled."""
        del prompt
        raise DatasetGeneratorError(
            "Dataset generation is disabled. Set DATASET_GENERATOR_MODE=gemini "
            "and configure DATASET_GENERATOR_API_KEY to enable it."
        )


class MockDatasetGenerator:
    """Deterministic provider for tests and offline development."""

    def __init__(self, response_text: str = "[]") -> None:
        self.response_text = response_text

    def generate_text(self, prompt: str) -> str:
        """Return the configured mock response."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return self.response_text


class GeminiDatasetGenerator:
    """Dataset authoring provider backed by the Gemini API."""

    def __init__(
        self,
        api_key: str = DATASET_GENERATOR_API_KEY,
        model: str = DATASET_GENERATOR_MODEL,
        timeout_seconds: float = DATASET_GENERATOR_TIMEOUT_SECONDS,
        max_output_tokens: int = DATASET_GENERATOR_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise DatasetGeneratorError(
                "DATASET_GENERATOR_API_KEY must be configured for Gemini generation"
            )
        if not isinstance(model, str) or not model.strip():
            raise DatasetGeneratorError("DATASET_GENERATOR_MODEL must be configured")
        if max_output_tokens <= 0:
            raise DatasetGeneratorError(
                "DATASET_GENERATOR_MAX_OUTPUT_TOKENS must be positive"
            )

        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    @property
    def endpoint_url(self) -> str:
        """Return the Gemini generateContent endpoint for the configured model."""
        normalized_model = self.model.strip()
        model_path = (
            normalized_model
            if normalized_model.startswith("models/")
            else f"models/{normalized_model}"
        )
        return f"{GEMINI_GENERATE_CONTENT_BASE_URL}/{model_path}:generateContent"

    def generate_text(self, prompt: str) -> str:
        """Generate dataset-authoring text through the Gemini API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt.strip(),
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "thinkingConfig": {
                    "thinkingBudget": 0,
                },
            },
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DatasetGeneratorError(
                f"Gemini dataset generation request failed: {exc}"
            ) from exc

        output_text = extract_response_text(data)
        if not output_text:
            raise DatasetGeneratorError("Gemini response did not include output text")
        return output_text


def get_dataset_generator_provider(
    mode: str | None = None,
) -> DatasetGeneratorProvider:
    """Return the configured dataset authoring provider."""
    normalized_mode = (mode or DATASET_GENERATOR_MODE).strip().lower()
    if normalized_mode == "disabled":
        return DisabledDatasetGenerator()
    if normalized_mode == "mock":
        return MockDatasetGenerator()
    if normalized_mode == "gemini":
        return GeminiDatasetGenerator(
            api_key=DATASET_GENERATOR_API_KEY,
            model=DATASET_GENERATOR_MODEL,
            timeout_seconds=DATASET_GENERATOR_TIMEOUT_SECONDS,
            max_output_tokens=DATASET_GENERATOR_MAX_OUTPUT_TOKENS,
        )

    raise DatasetGeneratorError(
        "DATASET_GENERATOR_MODE must be one of: disabled, mock, gemini"
    )


def extract_response_text(data: dict[str, Any]) -> str:
    """Extract text from common Gemini response shapes."""
    direct_text = data.get("output_text")
    if isinstance(direct_text, str):
        return direct_text.strip()

    candidates = data.get("candidates")
    if isinstance(candidates, list):
        parts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            content_parts = content.get("parts")
            if not isinstance(content_parts, list):
                continue
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(part.strip() for part in parts if part.strip()).strip()

    output = data.get("output")
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(part.strip() for part in parts if part.strip()).strip()
