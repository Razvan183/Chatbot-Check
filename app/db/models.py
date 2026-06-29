"""SQLAlchemy models for Chatbot Check's persistent data."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utc_now() -> datetime:
    """Return the current time in UTC for database timestamp defaults."""
    return datetime.now(timezone.utc)


class Document(Base):
    """A source file ingested into Chatbot Check."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    document_type: Mapped[str] = mapped_column(String(50))
    source_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """A searchable section of an ingested document."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_key: Mapped[str | None] = mapped_column(String(300), unique=True, index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class ChatbotVersion(Base):
    """A named set of RAG and generation settings."""

    __tablename__ = "chatbot_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), default="mock")
    embedding_model: Mapped[str] = mapped_column(
        String(255),
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    chunk_size: Mapped[int] = mapped_column(Integer, default=500)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)
    top_k: Mapped[int] = mapped_column(Integer, default=3)
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    prompt_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    chat_logs: Mapped[list["ChatLog"]] = relationship(back_populates="chatbot_version")
    eval_runs: Mapped[list["EvalRun"]] = relationship(back_populates="chatbot_version")


class RAGConnectorConfig(Base):
    """Persisted RAG target connector configuration."""

    __tablename__ = "rag_connector_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    connector_type: Mapped[str] = mapped_column(String(50), default="internal")
    http_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    timeout_seconds: Mapped[float] = mapped_column(Float, default=60)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ChatLog(Base):
    """One question and answer produced by a chatbot version."""

    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    chatbot_version_id: Mapped[int] = mapped_column(
        ForeignKey("chatbot_versions.id"),
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[str] = mapped_column(Text, default="[]")
    citations: Mapped[str] = mapped_column(Text, default="[]")
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    chatbot_version: Mapped["ChatbotVersion"] = relationship(back_populates="chat_logs")


class EvalDataset(Base):
    """A collection of evaluation questions for one domain."""

    __tablename__ = "eval_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    cases: Mapped[list["EvalCase"]] = relationship(
        back_populates="eval_dataset",
        cascade="all, delete-orphan",
    )
    runs: Mapped[list["EvalRun"]] = relationship(back_populates="eval_dataset")


class EvalCase(Base):
    """One expected chatbot behavior within an evaluation dataset."""

    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id"),
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_chunk_ids: Mapped[str] = mapped_column(Text, default="[]")
    expected_chunk_keys: Mapped[str] = mapped_column(Text, default="[]")
    question_type: Mapped[str] = mapped_column(String(50))
    difficulty: Mapped[str] = mapped_column(String(50))
    should_be_answerable: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    eval_dataset: Mapped["EvalDataset"] = relationship(back_populates="cases")
    results: Mapped[list["EvalResult"]] = relationship(back_populates="eval_case")


class DraftEvalDataset(Base):
    """LLM-generated evaluation dataset awaiting human review."""

    __tablename__ = "draft_eval_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), default="draft")
    requested_case_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    published_eval_dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("eval_datasets.id"),
        nullable=True,
    )

    cases: Mapped[list["DraftEvalCase"]] = relationship(
        back_populates="draft_dataset",
        cascade="all, delete-orphan",
    )


class DraftEvalCase(Base):
    """One candidate eval case generated by an authoring LLM."""

    __tablename__ = "draft_eval_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    draft_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("draft_eval_datasets.id"),
        index=True,
    )
    case_uid: Mapped[str] = mapped_column(String(50))
    question: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_chunk_keys: Mapped[str] = mapped_column(Text, default="[]")
    question_type: Mapped[str] = mapped_column(String(50))
    difficulty: Mapped[str] = mapped_column(String(50))
    should_be_answerable: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    draft_dataset: Mapped["DraftEvalDataset"] = relationship(back_populates="cases")


class EvalRun(Base):
    """One execution of a dataset against a chatbot version."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("eval_datasets.id"),
        index=True,
    )
    chatbot_version_id: Mapped[int] = mapped_column(
        ForeignKey("chatbot_versions.id"),
        index=True,
    )
    run_name: Mapped[str] = mapped_column(String(150))
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    eval_dataset: Mapped["EvalDataset"] = relationship(back_populates="runs")
    chatbot_version: Mapped["ChatbotVersion"] = relationship(back_populates="eval_runs")
    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="eval_run",
        cascade="all, delete-orphan",
    )


class EvalResult(Base):
    """Metrics and failure details for one evaluated question."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), index=True)
    eval_case_id: Mapped[int] = mapped_column(ForeignKey("eval_cases.id"), index=True)
    generated_answer: Mapped[str] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_chunk_keys: Mapped[str] = mapped_column(Text, default="[]")
    expected_chunk_hit: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_presence: Mapped[float | None] = mapped_column(Float, nullable=True)
    refusal_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    numeric_consistency: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_keyword_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    overall_case_score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    failure_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    eval_run: Mapped["EvalRun"] = relationship(back_populates="results")
    eval_case: Mapped["EvalCase"] = relationship(back_populates="results")
