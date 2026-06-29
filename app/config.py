"""Central configuration for Chatbot Check."""

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


APP_NAME = os.getenv("APP_NAME", "Chatbot Check")
APP_SERVICE_NAME = os.getenv("APP_SERVICE_NAME", "chatbot-check-api")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot_check.db")
DATA_DIR = os.getenv("DATA_DIR", "data")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
USE_MOCK_LLM = get_bool_env("USE_MOCK_LLM", True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
RAG_CONNECTOR = os.getenv("RAG_CONNECTOR", "internal")
HTTP_RAG_URL = os.getenv("HTTP_RAG_URL", "")
HTTP_RAG_TIMEOUT_SECONDS = float(os.getenv("HTTP_RAG_TIMEOUT_SECONDS", "60"))
DATASET_GENERATOR_MODE = os.getenv("DATASET_GENERATOR_MODE", "disabled")
DATASET_GENERATOR_MODEL = os.getenv("DATASET_GENERATOR_MODEL", "gemini-2.5-flash")
DATASET_GENERATOR_API_KEY = os.getenv("DATASET_GENERATOR_API_KEY", "")
DATASET_GENERATOR_TIMEOUT_SECONDS = float(
    os.getenv("DATASET_GENERATOR_TIMEOUT_SECONDS", "90")
)
DATASET_GENERATOR_MAX_OUTPUT_TOKENS = int(
    os.getenv("DATASET_GENERATOR_MAX_OUTPUT_TOKENS", "4000")
)
