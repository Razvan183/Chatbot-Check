"""Pydantic response schemas for the EvalForge API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    """Public metadata for an ingested document."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    document_type: str
    source_path: str
    status: str
    num_chunks: int
    created_at: datetime


class DocumentChunkResponse(BaseModel):
    """Public content and metadata for one document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    chunk_text: str
    section_title: str | None
    page_number: int | None
    created_at: datetime
