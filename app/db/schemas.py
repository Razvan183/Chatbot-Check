"""Pydantic response schemas for the Chatbot Check API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    chunk_key: str | None
    chunk_text: str
    section_title: str | None
    page_number: int | None
    created_at: datetime


class ChatRequest(BaseModel):
    """Question payload accepted by the chat endpoint."""

    question: str
    chatbot_version_id: int


class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk details returned with a chatbot answer."""

    chunk_id: int
    chunk_key: str
    document_id: int
    filename: str
    chunk_text: str
    score: float


class ChatResponse(BaseModel):
    """Answer payload returned by the chat endpoint."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    citations: list[int]
    latency_ms: int


class ChatbotVersionResponse(BaseModel):
    """Public settings for one chatbot version."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    prompt_template: str | None
    created_at: datetime


class ChatbotVersionCreateRequest(BaseModel):
    """Runtime settings used to create a tunable chatbot version."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    model_name: str = Field(default="mock", min_length=1, max_length=100)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        max_length=255,
    )
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=3, gt=0, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    prompt_template: str | None = None


class EvaluationDatasetResponse(BaseModel):
    """Public metadata for an evaluation dataset."""

    id: int
    name: str
    description: str | None
    domain: str
    case_count: int
    created_at: datetime


class DraftEvalDatasetCreateRequest(BaseModel):
    """Request to generate a draft evaluation dataset from ingested chunks."""

    dataset_name: str = Field(min_length=1, max_length=150)
    description: str | None = None
    domain: str = Field(default="company_policy", min_length=1, max_length=100)
    case_count: int = Field(default=10, ge=1, le=100)
    case_mix: dict[str, int] | None = None


class DraftEvalCaseResponse(BaseModel):
    """Human-reviewable candidate eval case."""

    id: int
    case_uid: str
    question: str
    expected_answer: str | None
    expected_chunk_keys: list[str]
    question_type: str
    difficulty: str
    should_be_answerable: bool
    confidence: float | None
    status: str
    reviewer_notes: str | None
    created_at: datetime
    updated_at: datetime


class DraftEvalDatasetResponse(BaseModel):
    """Draft dataset with review counts and candidate cases."""

    id: int
    name: str
    description: str | None
    domain: str
    status: str
    requested_case_count: int
    draft_case_count: int
    approved_case_count: int
    rejected_case_count: int
    published_eval_dataset_id: int | None
    created_at: datetime
    cases: list[DraftEvalCaseResponse]


class DraftEvalCaseReviewRequest(BaseModel):
    """Human review update for one generated draft case."""

    status: str = Field(pattern="^(draft|approved|rejected)$")
    question: str | None = None
    expected_answer: str | None = None
    expected_chunk_keys: list[str] | None = None
    question_type: str | None = None
    difficulty: str | None = None
    should_be_answerable: bool | None = None
    reviewer_notes: str | None = None


class DraftEvalDatasetPublishResponse(BaseModel):
    """Result of publishing approved draft cases into an official dataset."""

    eval_dataset_id: int
    case_count: int
    status: str


class EvaluationRunCreateRequest(BaseModel):
    """Payload used to start an evaluation run."""

    eval_dataset_id: int
    chatbot_version_id: int
    run_name: str | None = None


class EvaluationRunResponse(BaseModel):
    """Summary of one evaluation run."""

    id: int
    eval_dataset_id: int
    eval_dataset_name: str
    chatbot_version_id: int
    chatbot_version_name: str
    run_name: str
    overall_score: float | None
    status: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    created_at: datetime
    completed_at: datetime | None


class EvaluationRunStartedResponse(BaseModel):
    """Compact response returned after an evaluation run completes."""

    eval_run_id: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: float
    status: str


class EvaluationResultResponse(BaseModel):
    """Stored metrics and answer details for one evaluated case."""

    id: int
    eval_run_id: int
    eval_case_id: int
    question: str
    expected_answer: str | None
    generated_answer: str
    retrieved_chunk_ids: list[int]
    retrieved_chunk_keys: list[str]
    expected_chunk_hit: float | None
    citation_presence: float | None
    refusal_score: float | None
    numeric_consistency: float | None
    answer_keyword_score: float | None
    hallucination_flag: bool
    overall_case_score: float
    passed: bool
    failure_type: str | None
    failure_reason: str | None
    created_at: datetime


class EvaluationMetricScoreResponse(BaseModel):
    """Aggregated score for one evaluation quality dimension."""

    name: str
    label: str
    score: float | None
    measured_cases: int


class EvaluationFailureSummaryResponse(BaseModel):
    """Failure count for one run failure category."""

    failure_type: str
    count: int


class EvaluationTuningRecommendationResponse(BaseModel):
    """Actionable parameter recommendation derived from a scorecard."""

    parameter: str
    current_value: str
    suggested_value: str
    reason: str


class EvaluationScorecardResponse(BaseModel):
    """Single-run RAG quality scorecard."""

    run: EvaluationRunResponse
    metric_scores: list[EvaluationMetricScoreResponse]
    failure_summary: list[EvaluationFailureSummaryResponse]
    recommendations: list[EvaluationTuningRecommendationResponse]


class EvaluationFailureBreakdownResponse(BaseModel):
    """Failure-type counts for one evaluation run."""

    failure_type: str
    baseline_count: int
    candidate_count: int
    delta: int


class EvaluationCaseComparisonResponse(BaseModel):
    """Side-by-side score and failure details for one eval case."""

    eval_case_id: int
    question: str
    baseline_score: float
    candidate_score: float
    score_delta: float
    baseline_passed: bool
    candidate_passed: bool
    baseline_failure_type: str | None
    candidate_failure_type: str | None
    status: str


class EvaluationRunComparisonResponse(BaseModel):
    """Regression comparison between two evaluation runs."""

    baseline_run: EvaluationRunResponse
    candidate_run: EvaluationRunResponse
    overall_score_delta: float | None
    passed_cases_delta: int
    failed_cases_delta: int
    fixed_cases: int
    new_failures: int
    improved_cases: int
    regressed_cases: int
    unchanged_cases: int
    failure_breakdown: list[EvaluationFailureBreakdownResponse]
    case_comparisons: list[EvaluationCaseComparisonResponse]
