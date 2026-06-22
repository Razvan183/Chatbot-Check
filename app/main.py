"""FastAPI entry point for the EvalForge backend."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.config import APP_NAME
from app.db.database import create_database_tables


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize application resources during startup."""
    create_database_tables()
    yield

app = FastAPI(
    title=f"{APP_NAME} API",
    description="Backend API for evaluating RAG chatbot quality.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return a simple response confirming that the API is running."""
    return {
        "status": "ok",
        "service": "evalforge-api",
    }
