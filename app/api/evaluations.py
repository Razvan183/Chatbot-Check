"""API endpoints for evaluation datasets, runs, and results."""

from datetime import datetime, timezone
from html import escape
import json
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ChatbotVersion, EvalCase, EvalDataset, EvalResult, EvalRun
from app.db.schemas import (
    EvaluationCaseComparisonResponse,
    EvaluationDatasetResponse,
    EvaluationFailureBreakdownResponse,
    EvaluationFailureSummaryResponse,
    EvaluationMetricScoreResponse,
    EvaluationResultResponse,
    EvaluationRunCreateRequest,
    EvaluationRunComparisonResponse,
    EvaluationRunResponse,
    EvaluationScorecardResponse,
    EvaluationRunStartedResponse,
    EvaluationTuningRecommendationResponse,
)
from app.evaluation.runner import (
    EvaluationDatasetNotFoundError,
    EvaluationRunError,
    create_evaluation_run_record,
    execute_evaluation_run,
)
from app.rag.pipeline import ChatbotVersionNotFoundError


router = APIRouter(prefix="/evaluations", tags=["Evaluations"])
DatabaseSession = Annotated[Session, Depends(get_db)]


def _case_count(database_session: Session, dataset_id: int) -> int:
    """Return the number of cases in one dataset."""
    return int(
        database_session.scalar(
            select(func.count(EvalCase.id)).where(
                EvalCase.eval_dataset_id == dataset_id
            )
        )
        or 0
    )


def _run_counts(database_session: Session, run_id: int) -> tuple[int, int, int]:
    """Return total, passed, and failed result counts for one run."""
    statement = select(
        func.count(EvalResult.id),
        func.coalesce(
            func.sum(case((EvalResult.passed.is_(True), 1), else_=0)),
            0,
        ),
    ).where(EvalResult.eval_run_id == run_id)
    total_cases, passed_cases = database_session.execute(statement).one()
    total = int(total_cases or 0)
    passed = int(passed_cases or 0)
    return total, passed, total - passed


def _run_response(database_session: Session, eval_run: EvalRun) -> EvaluationRunResponse:
    """Build a public run summary with related names and result counts."""
    total_cases, passed_cases, failed_cases = _run_counts(database_session, eval_run.id)

    return EvaluationRunResponse(
        id=eval_run.id,
        eval_dataset_id=eval_run.eval_dataset_id,
        eval_dataset_name=eval_run.eval_dataset.name,
        chatbot_version_id=eval_run.chatbot_version_id,
        chatbot_version_name=eval_run.chatbot_version.name,
        run_name=eval_run.run_name,
        overall_score=eval_run.overall_score,
        status=eval_run.status,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        created_at=eval_run.created_at,
        completed_at=eval_run.completed_at,
    )


def _default_run_name(
    database_session: Session,
    eval_dataset_id: int,
    chatbot_version_id: int,
) -> str:
    """Build a readable default name for API-created runs."""
    dataset = database_session.get(EvalDataset, eval_dataset_id)
    if dataset is None:
        raise EvaluationDatasetNotFoundError("Evaluation dataset not found")

    chatbot_version = database_session.get(ChatbotVersion, chatbot_version_id)
    if chatbot_version is None:
        raise ChatbotVersionNotFoundError("Chatbot version not found")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{dataset.name} / {chatbot_version.name} / {timestamp}"


def _decode_int_list(raw_value: str) -> list[int]:
    """Decode a JSON list of integers stored in a text column."""
    if raw_value is None:
        return []
    decoded = json.loads(raw_value)
    if not isinstance(decoded, list):
        return []
    return [int(value) for value in decoded]


def _decode_str_list(raw_value: str) -> list[str]:
    """Decode a JSON list of strings stored in a text column."""
    if raw_value is None:
        return []
    decoded = json.loads(raw_value)
    if not isinstance(decoded, list):
        return []
    return [str(value) for value in decoded]


def _load_run_or_404(database_session: Session, eval_run_id: int) -> EvalRun:
    """Return one evaluation run or raise the shared 404 response."""
    eval_run = database_session.get(EvalRun, eval_run_id)
    if eval_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found",
        )
    return eval_run


def _results_by_case(
    database_session: Session,
    eval_run_id: int,
) -> dict[int, tuple[EvalResult, EvalCase]]:
    """Load result rows keyed by eval case id."""
    statement = (
        select(EvalResult, EvalCase)
        .join(EvalCase, EvalResult.eval_case_id == EvalCase.id)
        .where(EvalResult.eval_run_id == eval_run_id)
    )
    return {
        result.eval_case_id: (result, eval_case)
        for result, eval_case in database_session.execute(statement)
    }


def _failure_counts(results: list[EvalResult]) -> dict[str, int]:
    """Count non-passing results by failure type."""
    counts: dict[str, int] = {}
    for result in results:
        if result.passed:
            continue
        failure_type = result.failure_type or "unknown_failure"
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return counts


def _comparison_status(baseline: EvalResult, candidate: EvalResult) -> str:
    """Classify the case-level movement between two results."""
    if not baseline.passed and candidate.passed:
        return "fixed"
    if baseline.passed and not candidate.passed:
        return "new_failure"

    score_delta = round(candidate.overall_case_score - baseline.overall_case_score, 4)
    if score_delta > 0:
        return "improved"
    if score_delta < 0:
        return "regressed"
    return "unchanged"


def _average_metric(results: list[EvalResult], metric_name: str) -> tuple[float | None, int]:
    """Return the average non-null metric value and number of measured cases."""
    values = [
        float(value)
        for result in results
        if (value := getattr(result, metric_name)) is not None
    ]
    if not values:
        return None, 0
    return round(sum(values) / len(values), 4), len(values)


def _metric_scores(results: list[EvalResult]) -> list[EvaluationMetricScoreResponse]:
    """Build the main quality dimensions for a single RAG run."""
    metric_definitions = [
        ("expected_chunk_hit", "Retrieval"),
        ("citation_presence", "Citation"),
        ("refusal_score", "Refusal"),
        ("numeric_consistency", "Numeric consistency"),
        ("answer_keyword_score", "Answer relevance"),
    ]
    scores = []
    for metric_name, label in metric_definitions:
        score, measured_cases = _average_metric(results, metric_name)
        scores.append(
            EvaluationMetricScoreResponse(
                name=metric_name,
                label=label,
                score=score,
                measured_cases=measured_cases,
            )
        )
    return scores


def _failure_summary(results: list[EvalResult]) -> list[EvaluationFailureSummaryResponse]:
    """Return failure counts for one evaluation run."""
    counts = _failure_counts(results)
    return [
        EvaluationFailureSummaryResponse(failure_type=failure_type, count=count)
        for failure_type, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _score_by_name(
    metric_scores: list[EvaluationMetricScoreResponse],
) -> dict[str, float | None]:
    """Index metric score values by internal metric name."""
    return {metric.name: metric.score for metric in metric_scores}


def _recommendations(
    eval_run: EvalRun,
    metric_scores: list[EvaluationMetricScoreResponse],
    failure_summary: list[EvaluationFailureSummaryResponse],
) -> list[EvaluationTuningRecommendationResponse]:
    """Derive practical tuning recommendations from scorecard signals."""
    scores = _score_by_name(metric_scores)
    failures = {failure.failure_type: failure.count for failure in failure_summary}
    version = eval_run.chatbot_version
    recommendations: list[EvaluationTuningRecommendationResponse] = []

    def add(parameter: str, current: object, suggested: object, reason: str) -> None:
        recommendations.append(
            EvaluationTuningRecommendationResponse(
                parameter=parameter,
                current_value=str(current),
                suggested_value=str(suggested),
                reason=reason,
            )
        )

    retrieval_score = scores.get("expected_chunk_hit")
    if retrieval_score is not None and retrieval_score < 0.85:
        add(
            "top_k",
            version.top_k,
            min(version.top_k + 2, 20),
            "Retrieval score is low; retrieving more chunks may surface missing evidence.",
        )

    if failures.get("retrieval_failure", 0) > 0:
        add(
            "retrieval_strategy",
            "semantic_only",
            "hybrid_search_or_reranking",
            "Some expected evidence was not retrieved; add keyword fallback or reranking before changing the generator.",
        )

    citation_score = scores.get("citation_presence")
    if citation_score is not None and citation_score < 0.9:
        add(
            "prompt_template",
            "current",
            "stricter citation instructions",
            "Citation score is below target; require every factual sentence to cite a retrieved chunk.",
        )

    refusal_score = scores.get("refusal_score")
    if refusal_score is not None and refusal_score < 0.9:
        add(
            "prompt_template",
            "current",
            "stricter refusal instructions",
            "The system answered at least one unanswerable question; make refusal behavior explicit.",
        )

    if failures.get("answered_unanswerable_question", 0) > 0:
        add(
            "retrieval_score_threshold",
            "not configured",
            "add minimum evidence threshold",
            "Unanswerable questions should be blocked when retrieved evidence is weak or unrelated.",
        )

    numeric_score = scores.get("numeric_consistency")
    if numeric_score is not None and numeric_score < 1.0:
        add(
            "temperature",
            version.temperature,
            0.0,
            "Unsupported numbers appeared in answers; use deterministic generation and stricter numeric grounding.",
        )

    if failures.get("missing_citation", 0) > 0:
        add(
            "citation_validator",
            "not configured",
            "enabled",
            "Missing citations can be caught after generation before returning the answer.",
        )

    if not recommendations:
        add(
            "release_status",
            "current",
            "ready_for_review",
            "No dominant scoring weakness was detected; inspect remaining failed cases manually.",
        )

    return recommendations[:6]


def _format_percent(value: float | None) -> str:
    """Format a stored score as a report-friendly percentage."""
    if value is None:
        return "n/a"
    return f"{round(value * 100)}%"


def _format_delta(value: float | int | None, percentage: bool = False) -> str:
    """Format numeric deltas with a visible sign."""
    if value is None:
        return "n/a"
    scaled_value = round(value * 100) if percentage else value
    sign = "+" if scaled_value > 0 else ""
    suffix = "%" if percentage else ""
    return f"{sign}{scaled_value}{suffix}"


def _movement_label(status_value: str) -> str:
    """Return a compact human label for a case movement status."""
    labels = {
        "fixed": "Fixed",
        "new_failure": "New failure",
        "improved": "Improved",
        "regressed": "Regressed",
        "unchanged": "Unchanged",
    }
    return labels.get(status_value, status_value.replace("_", " ").title())


def _render_metric(label: str, value: str) -> str:
    """Render one report metric tile."""
    return (
        '<div class="metric">'
        f"<strong>{escape(value)}</strong>"
        f"<span>{escape(label)}</span>"
        "</div>"
    )


def _render_comparison_report(comparison: EvaluationRunComparisonResponse) -> str:
    """Render a standalone HTML comparison report."""
    baseline = comparison.baseline_run
    candidate = comparison.candidate_run
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    failure_rows = "\n".join(
        "<tr>"
        f"<td>{escape(item.failure_type)}</td>"
        f"<td>{item.baseline_count}</td>"
        f"<td>{item.candidate_count}</td>"
        f"<td>{_format_delta(item.delta)}</td>"
        "</tr>"
        for item in comparison.failure_breakdown
    ) or '<tr><td colspan="4">No failures in either run.</td></tr>'

    case_rows = "\n".join(
        "<tr>"
        f"<td>{case.eval_case_id}</td>"
        f"<td>{escape(_movement_label(case.status))}</td>"
        f"<td>{escape(case.question)}</td>"
        f"<td>{_format_percent(case.baseline_score)}</td>"
        f"<td>{_format_percent(case.candidate_score)}</td>"
        f"<td>{_format_delta(case.score_delta, percentage=True)}</td>"
        f"<td>{escape(case.baseline_failure_type or 'none')}</td>"
        f"<td>{escape(case.candidate_failure_type or 'none')}</td>"
        "</tr>"
        for case in comparison.case_comparisons
    )

    summary_metrics = "".join(
        [
            _render_metric("Score delta", _format_delta(comparison.overall_score_delta, percentage=True)),
            _render_metric("Passed delta", _format_delta(comparison.passed_cases_delta)),
            _render_metric("Failed delta", _format_delta(comparison.failed_cases_delta)),
            _render_metric("Fixed cases", str(comparison.fixed_cases)),
            _render_metric("New failures", str(comparison.new_failures)),
            _render_metric("Regressed cases", str(comparison.regressed_cases)),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RAG Evaluation Comparison Report</title>
    <style>
      :root {{
        --bg: #f6f7f4;
        --panel: #ffffff;
        --ink: #1c2321;
        --muted: #5d6865;
        --line: #dfe4dd;
        --accent: #0b6b5e;
        --red: #b6423c;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: Inter, "Segoe UI", Arial, sans-serif;
        line-height: 1.5;
      }}
      main {{
        width: min(1180px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 48px;
      }}
      header {{
        display: grid;
        gap: 12px;
        padding-bottom: 22px;
        border-bottom: 1px solid var(--line);
      }}
      h1, h2, p {{ margin: 0; }}
      h1 {{ font-size: 2rem; }}
      h2 {{ margin-top: 28px; font-size: 1.15rem; }}
      .muted {{ color: var(--muted); }}
      .run-grid, .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-top: 18px;
      }}
      .run-card, .metric {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
      }}
      .metric strong {{
        display: block;
        color: var(--accent);
        font-size: 1.7rem;
      }}
      .metric span, .run-card span {{
        color: var(--muted);
        font-size: 0.86rem;
        font-weight: 700;
      }}
      table {{
        width: 100%;
        margin-top: 14px;
        border-collapse: collapse;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
      }}
      th, td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}
      th {{
        background: #e8eee9;
        font-size: 0.82rem;
        text-transform: uppercase;
      }}
      tr:last-child td {{ border-bottom: 0; }}
      @media print {{
        body {{ background: white; }}
        main {{ width: 100%; padding: 0; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header>
        <p class="muted">Chatbot Check report generated {escape(generated_at)}</p>
        <h1>{escape(baseline.chatbot_version_name)} vs {escape(candidate.chatbot_version_name)}</h1>
        <p>{escape(baseline.eval_dataset_name)} comparison for completed RAG evaluation runs.</p>
      </header>

      <section>
        <h2>Runs</h2>
        <div class="run-grid">
          <article class="run-card">
            <span>Baseline</span>
            <h3>{escape(baseline.run_name)}</h3>
            <p>{_format_percent(baseline.overall_score)} score, {baseline.passed_cases}/{baseline.total_cases} passed</p>
          </article>
          <article class="run-card">
            <span>Candidate</span>
            <h3>{escape(candidate.run_name)}</h3>
            <p>{_format_percent(candidate.overall_score)} score, {candidate.passed_cases}/{candidate.total_cases} passed</p>
          </article>
        </div>
      </section>

      <section>
        <h2>Summary</h2>
        <div class="metric-grid">{summary_metrics}</div>
      </section>

      <section>
        <h2>Failure Types</h2>
        <table>
          <thead>
            <tr><th>Failure type</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>
          </thead>
          <tbody>{failure_rows}</tbody>
        </table>
      </section>

      <section>
        <h2>Case-Level Results</h2>
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Status</th>
              <th>Question</th>
              <th>Baseline</th>
              <th>Candidate</th>
              <th>Delta</th>
              <th>Baseline failure</th>
              <th>Candidate failure</th>
            </tr>
          </thead>
          <tbody>{case_rows}</tbody>
        </table>
      </section>
    </main>
  </body>
</html>"""


@router.get("/datasets", response_model=list[EvaluationDatasetResponse])
def list_evaluation_datasets(
    database_session: DatabaseSession,
) -> list[EvaluationDatasetResponse]:
    """Return evaluation datasets ordered by name."""
    datasets = list(
        database_session.scalars(select(EvalDataset).order_by(EvalDataset.name))
    )
    return [
        EvaluationDatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            domain=dataset.domain,
            case_count=_case_count(database_session, dataset.id),
            created_at=dataset.created_at,
        )
        for dataset in datasets
    ]


@router.get("/runs", response_model=list[EvaluationRunResponse])
def list_evaluation_runs(
    database_session: DatabaseSession,
) -> list[EvaluationRunResponse]:
    """Return evaluation runs ordered newest first."""
    statement = select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
    return [
        _run_response(database_session, eval_run)
        for eval_run in database_session.scalars(statement)
    ]


@router.post(
    "/runs",
    response_model=EvaluationRunStartedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_evaluation_run(
    request: EvaluationRunCreateRequest,
    background_tasks: BackgroundTasks,
    database_session: DatabaseSession,
) -> dict:
    """Queue a dataset run against a chatbot version and persist the results."""
    try:
        run_name = request.run_name
        if run_name is None or not run_name.strip():
            run_name = _default_run_name(
                database_session,
                request.eval_dataset_id,
                request.chatbot_version_id,
            )

        eval_run_id = create_evaluation_run_record(
            eval_dataset_id=request.eval_dataset_id,
            chatbot_version_id=request.chatbot_version_id,
            run_name=run_name,
        )
        background_tasks.add_task(execute_evaluation_run, eval_run_id)
        total_cases = _case_count(database_session, request.eval_dataset_id)
        return {
            "eval_run_id": eval_run_id,
            "total_cases": total_cases,
            "passed_cases": 0,
            "failed_cases": 0,
            "overall_score": 0.0,
            "status": "pending",
        }
    except (EvaluationDatasetNotFoundError, ChatbotVersionNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (EvaluationRunError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/runs/{eval_run_id}/scorecard",
    response_model=EvaluationScorecardResponse,
)
def get_evaluation_scorecard(
    eval_run_id: int,
    database_session: DatabaseSession,
) -> EvaluationScorecardResponse:
    """Return the primary single-run RAG quality scorecard."""
    eval_run = _load_run_or_404(database_session, eval_run_id)
    statement = (
        select(EvalResult)
        .where(EvalResult.eval_run_id == eval_run_id)
        .order_by(EvalResult.id)
    )
    results = list(database_session.scalars(statement))
    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation run has no case results",
        )

    metric_scores = _metric_scores(results)
    failure_summary = _failure_summary(results)
    return EvaluationScorecardResponse(
        run=_run_response(database_session, eval_run),
        metric_scores=metric_scores,
        failure_summary=failure_summary,
        recommendations=_recommendations(eval_run, metric_scores, failure_summary),
    )


@router.get(
    "/runs/{baseline_run_id}/compare/{candidate_run_id}",
    response_model=EvaluationRunComparisonResponse,
)
def compare_evaluation_runs(
    baseline_run_id: int,
    candidate_run_id: int,
    database_session: DatabaseSession,
) -> EvaluationRunComparisonResponse:
    """Compare two runs from the same dataset and return regression deltas."""
    baseline_run = _load_run_or_404(database_session, baseline_run_id)
    candidate_run = _load_run_or_404(database_session, candidate_run_id)

    if baseline_run.eval_dataset_id != candidate_run.eval_dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation runs must belong to the same dataset",
        )

    baseline_results_by_case = _results_by_case(database_session, baseline_run.id)
    candidate_results_by_case = _results_by_case(database_session, candidate_run.id)
    shared_case_ids = sorted(
        set(baseline_results_by_case) & set(candidate_results_by_case)
    )

    if not shared_case_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation runs do not have comparable case results",
        )

    case_comparisons: list[EvaluationCaseComparisonResponse] = []
    status_counts = {
        "fixed": 0,
        "new_failure": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged": 0,
    }

    for eval_case_id in shared_case_ids:
        baseline_result, eval_case = baseline_results_by_case[eval_case_id]
        candidate_result, _ = candidate_results_by_case[eval_case_id]
        score_delta = round(
            candidate_result.overall_case_score - baseline_result.overall_case_score,
            4,
        )
        movement = _comparison_status(baseline_result, candidate_result)
        status_counts[movement] += 1

        case_comparisons.append(
            EvaluationCaseComparisonResponse(
                eval_case_id=eval_case_id,
                question=eval_case.question,
                baseline_score=baseline_result.overall_case_score,
                candidate_score=candidate_result.overall_case_score,
                score_delta=score_delta,
                baseline_passed=baseline_result.passed,
                candidate_passed=candidate_result.passed,
                baseline_failure_type=baseline_result.failure_type,
                candidate_failure_type=candidate_result.failure_type,
                status=movement,
            )
        )

    baseline_counts = _failure_counts(
        [result for result, _ in baseline_results_by_case.values()]
    )
    candidate_counts = _failure_counts(
        [result for result, _ in candidate_results_by_case.values()]
    )
    failure_breakdown = [
        EvaluationFailureBreakdownResponse(
            failure_type=failure_type,
            baseline_count=baseline_counts.get(failure_type, 0),
            candidate_count=candidate_counts.get(failure_type, 0),
            delta=(
                candidate_counts.get(failure_type, 0)
                - baseline_counts.get(failure_type, 0)
            ),
        )
        for failure_type in sorted(set(baseline_counts) | set(candidate_counts))
    ]

    baseline_response = _run_response(database_session, baseline_run)
    candidate_response = _run_response(database_session, candidate_run)
    overall_score_delta = None
    if (
        baseline_response.overall_score is not None
        and candidate_response.overall_score is not None
    ):
        overall_score_delta = round(
            candidate_response.overall_score - baseline_response.overall_score,
            4,
        )

    return EvaluationRunComparisonResponse(
        baseline_run=baseline_response,
        candidate_run=candidate_response,
        overall_score_delta=overall_score_delta,
        passed_cases_delta=(
            candidate_response.passed_cases - baseline_response.passed_cases
        ),
        failed_cases_delta=(
            candidate_response.failed_cases - baseline_response.failed_cases
        ),
        fixed_cases=status_counts["fixed"],
        new_failures=status_counts["new_failure"],
        improved_cases=status_counts["improved"],
        regressed_cases=status_counts["regressed"],
        unchanged_cases=status_counts["unchanged"],
        failure_breakdown=failure_breakdown,
        case_comparisons=case_comparisons,
    )


@router.get(
    "/runs/{baseline_run_id}/compare/{candidate_run_id}/report",
    response_class=HTMLResponse,
)
def get_evaluation_comparison_report(
    baseline_run_id: int,
    candidate_run_id: int,
    database_session: DatabaseSession,
) -> HTMLResponse:
    """Return a standalone HTML report for a run comparison."""
    comparison = compare_evaluation_runs(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        database_session=database_session,
    )
    return HTMLResponse(_render_comparison_report(comparison))


@router.get("/runs/{eval_run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run(
    eval_run_id: int,
    database_session: DatabaseSession,
) -> EvaluationRunResponse:
    """Return one evaluation run summary or a clear 404 response."""
    eval_run = _load_run_or_404(database_session, eval_run_id)
    return _run_response(database_session, eval_run)


@router.get(
    "/runs/{eval_run_id}/results",
    response_model=list[EvaluationResultResponse],
)
def list_evaluation_run_results(
    eval_run_id: int,
    database_session: DatabaseSession,
) -> list[EvaluationResultResponse]:
    """Return case-level results for one evaluation run."""
    _load_run_or_404(database_session, eval_run_id)

    statement = (
        select(EvalResult, EvalCase)
        .join(EvalCase, EvalResult.eval_case_id == EvalCase.id)
        .where(EvalResult.eval_run_id == eval_run_id)
        .order_by(EvalResult.id)
    )
    return [
        EvaluationResultResponse(
            id=result.id,
            eval_run_id=result.eval_run_id,
            eval_case_id=result.eval_case_id,
            question=eval_case.question,
            expected_answer=eval_case.expected_answer,
            generated_answer=result.generated_answer,
            retrieved_chunk_ids=_decode_int_list(result.retrieved_chunk_ids),
            retrieved_chunk_keys=_decode_str_list(result.retrieved_chunk_keys),
            expected_chunk_hit=result.expected_chunk_hit,
            citation_presence=result.citation_presence,
            refusal_score=result.refusal_score,
            numeric_consistency=result.numeric_consistency,
            answer_keyword_score=result.answer_keyword_score,
            hallucination_flag=result.hallucination_flag,
            overall_case_score=result.overall_case_score,
            passed=result.passed,
            failure_type=result.failure_type,
            failure_reason=result.failure_reason,
            created_at=result.created_at,
        )
        for result, eval_case in database_session.execute(statement)
    ]
