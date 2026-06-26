"""API endpoints for chatbot versions and runtime tuning settings."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ChatbotVersion
from app.db.schemas import ChatbotVersionCreateRequest, ChatbotVersionResponse


router = APIRouter(prefix="/chatbot-versions", tags=["Chatbot Versions"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=list[ChatbotVersionResponse])
def list_chatbot_versions(database_session: DatabaseSession) -> list[ChatbotVersion]:
    """Return chatbot versions ordered by name."""
    statement = select(ChatbotVersion).order_by(ChatbotVersion.name)
    return list(database_session.scalars(statement))


@router.post(
    "",
    response_model=ChatbotVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_chatbot_version(
    request: ChatbotVersionCreateRequest,
    database_session: DatabaseSession,
) -> ChatbotVersion:
    """Create a chatbot version from visible runtime tuning settings."""
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    chatbot_version = ChatbotVersion(
        name=request.name.strip(),
        description=request.description.strip() if request.description else None,
        model_name=request.model_name.strip(),
        embedding_model=request.embedding_model.strip(),
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        top_k=request.top_k,
        temperature=request.temperature,
        prompt_template=(
            request.prompt_template.strip()
            if request.prompt_template and request.prompt_template.strip()
            else None
        ),
    )
    database_session.add(chatbot_version)

    try:
        database_session.commit()
        database_session.refresh(chatbot_version)
    except IntegrityError as exc:
        database_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Chatbot version name already exists",
        ) from exc

    return chatbot_version
