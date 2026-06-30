"""Embedding helpers used by semantic retrieval."""

from functools import lru_cache
import hashlib
import re
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from app.config import DEFAULT_EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


FALLBACK_EMBEDDING_DIMENSIONS = 64
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    """Load the configured sentence-transformer model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL, local_files_only=True)


def build_fallback_embedding(text: str) -> list[float]:
    """Create a deterministic lexical embedding when the ML model is unavailable."""
    vector = np.zeros(FALLBACK_EMBEDDING_DIMENSIONS, dtype=float)
    tokens = TOKEN_PATTERN.findall(text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % FALLBACK_EMBEDDING_DIMENSIONS
        vector[index] += 1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()

    return (vector / norm).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create one numeric embedding for each supplied text."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain only strings")
    if not texts:
        return []

    try:
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float).tolist()
    except Exception:
        cache_clear = getattr(get_embedding_model, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
        return [build_fallback_embedding(text) for text in texts]



def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Return cosine similarity for two equally sized one-dimensional vectors."""
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)

    if vector_a.ndim != 1 or vector_b.ndim != 1:
        raise ValueError("cosine similarity requires one-dimensional vectors")
    if vector_a.size == 0 or vector_b.size == 0:
        raise ValueError("cosine similarity requires non-empty vectors")
    if vector_a.shape != vector_b.shape:
        raise ValueError("cosine similarity requires vectors of equal length")

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / denominator
    return float(np.clip(similarity, -1.0, 1.0))
