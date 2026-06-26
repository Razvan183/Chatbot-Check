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


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


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
            "Dataset generation is disabled. Set DATASET_GENERATOR_MODE=openai "
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


class OpenAIDatasetGenerator:
    """Dataset authoring provider backed by the OpenAI Responses API."""

    def __init__(
        self,
        api_key: str = DATASET_GENERATOR_API_KEY,
        model: str = DATASET_GENERATOR_MODEL,
        timeout_seconds: float = DATASET_GENERATOR_TIMEOUT_SECONDS,
        max_output_tokens: int = DATASET_GENERATOR_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise DatasetGeneratorError(
                "DATASET_GENERATOR_API_KEY must be configured for OpenAI generation"
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

    def generate_text(self, prompt: str) -> str:
        """Generate dataset-authoring text through the Responses API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        payload = {
            "model": self.model,
            "input": prompt.strip(),
            "max_output_tokens": self.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                OPENAI_RESPONSES_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DatasetGeneratorError(
                f"OpenAI dataset generation request failed: {exc}"
            ) from exc

        output_text = extract_response_text(data)
        if not output_text:
            raise DatasetGeneratorError("OpenAI response did not include output text")
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
    if normalized_mode == "openai":
        return OpenAIDatasetGenerator()

    raise DatasetGeneratorError(
        "DATASET_GENERATOR_MODE must be one of: disabled, mock, openai"
    )


def extract_response_text(data: dict[str, Any]) -> str:
    """Extract text from common Responses API response shapes."""
    direct_text = data.get("output_text")
    if isinstance(direct_text, str):
        return direct_text.strip()

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
