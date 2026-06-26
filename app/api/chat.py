"""API endpoint for asking the RAG chatbot questions."""

from fastapi import APIRouter, HTTPException, status

from app.db.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import ChatbotVersionNotFoundError, answer_question


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def create_chat_response(request: ChatRequest) -> dict:
    """Answer a user question through the RAG pipeline."""
    try:
        return answer_question(
            request.question,
            chatbot_version_id=request.chatbot_version_id,
        )
    except ChatbotVersionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
