"""Tests for embedding generation and vector similarity."""

import numpy as np
import pytest

from app.ingestion import embeddings


class FakeEmbeddingModel:
    """Small deterministic replacement for the external embedding model."""

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert show_progress_bar is False
        return np.asarray(
            [[float(len(text)), float(text.count(" "))] for text in texts]
        )


def test_embed_texts_returns_one_embedding_per_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeEmbeddingModel(),
    )

    result = embeddings.embed_texts(["policy", "remote work"])

    assert result == [[6.0, 0.0], [11.0, 1.0]]


def test_embed_texts_skips_model_loading_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        raise AssertionError("The model should not be loaded")

    monkeypatch.setattr(embeddings, "get_embedding_model", fail_if_called)

    assert embeddings.embed_texts([]) == []


@pytest.mark.parametrize(
    "texts",
    [
        "policy",
        ("policy",),
        ["policy", 123],
    ],
)
def test_embed_texts_rejects_invalid_input(texts) -> None:
    with pytest.raises(TypeError):
        embeddings.embed_texts(texts)


def test_get_embedding_model_uses_configured_name_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_names: list[str] = []
    sentinel_model = object()

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str):
            loaded_model_names.append(model_name)
            return sentinel_model

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    embeddings.get_embedding_model.cache_clear()

    try:
        first_model = embeddings.get_embedding_model()
        second_model = embeddings.get_embedding_model()
    finally:
        embeddings.get_embedding_model.cache_clear()

    assert first_model is sentinel_model
    assert second_model is sentinel_model
    assert loaded_model_names == [embeddings.DEFAULT_EMBEDDING_MODEL]


def test_cosine_similarity_for_matching_vectors() -> None:
    assert embeddings.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    assert embeddings.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_returns_zero_for_zero_vector() -> None:
    assert embeddings.cosine_similarity([0, 0], [1, 1]) == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([], []),
        ([1, 2], [1]),
        ([[1, 2]], [[1, 2]]),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(a, b) -> None:
    with pytest.raises(ValueError):
        embeddings.cosine_similarity(a, b)
