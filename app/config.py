"""Central configuration for EvalForge."""

import os

from dotenv import load_dotenv


# Load values from a local .env file into the process environment.
# Existing operating-system environment variables keep priority.
load_dotenv()


def get_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable using common true values."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


APP_NAME = os.getenv("APP_NAME", "EvalForge")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evalforge.db")
DATA_DIR = os.getenv("DATA_DIR", "data")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
USE_MOCK_LLM = get_bool_env("USE_MOCK_LLM", True)
