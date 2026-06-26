"""FastAPI entry point for the Chatbot Check backend."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.evaluation_drafts import router as evaluation_drafts_router
from app.api.evaluations import router as evaluations_router
from app.api.versions import router as versions_router
from app.config import APP_NAME, APP_SERVICE_NAME
from app.db.database import create_database_tables


BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = BASE_DIR / "demo"


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

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(evaluation_drafts_router)
app.include_router(evaluations_router)
app.include_router(versions_router)

app.mount("/static", StaticFiles(directory=DEMO_DIR), name="static")


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Return a simple response confirming that the API is running."""
    return {
        "status": "ok",
        "service": APP_SERVICE_NAME,
    }


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    """Serve the browser-based demo interface."""
    return FileResponse(DEMO_DIR / "index.html")
