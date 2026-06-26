"""Run an evaluation dataset against a chatbot version."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


# Allow both module execution and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.database import SessionLocal, create_database_tables
from app.db.models import ChatbotVersion, EvalDataset
from app.evaluation.runner import run_evaluation


def _find_dataset_id(dataset_name: str) -> int:
    """Return a dataset id by name or raise a clear error."""
    with SessionLocal() as database_session:
        dataset_id = database_session.scalar(
            select(EvalDataset.id).where(EvalDataset.name == dataset_name)
        )
    if dataset_id is None:
        raise ValueError(f"Evaluation dataset not found: {dataset_name}")
    return dataset_id


def _find_chatbot_version_id(version_name: str) -> int:
    """Return a chatbot version id by name or raise a clear error."""
    with SessionLocal() as database_session:
        version_id = database_session.scalar(
            select(ChatbotVersion.id).where(ChatbotVersion.name == version_name)
        )
    if version_id is None:
        raise ValueError(f"Chatbot version not found: {version_name}")
    return version_id


def _default_run_name(dataset_name: str, version_name: str) -> str:
    """Build a readable default run name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{dataset_name} / {version_name} / {timestamp}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="HR Policy Eval Set",
        help="Evaluation dataset name to run.",
    )
    parser.add_argument(
        "--version-name",
        default="baseline_v1",
        help="Chatbot version name to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional display name for the evaluation run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected evaluation and print a compact summary."""
    args = parse_args()
    create_database_tables()

    dataset_id = _find_dataset_id(args.dataset_name)
    version_id = _find_chatbot_version_id(args.version_name)
    run_name = args.run_name or _default_run_name(args.dataset_name, args.version_name)

    summary = run_evaluation(
        eval_dataset_id=dataset_id,
        chatbot_version_id=version_id,
        run_name=run_name,
    )

    print(f"Evaluation run {summary['eval_run_id']} completed")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed cases: {summary['passed_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(f"Overall score: {summary['overall_score']}")


if __name__ == "__main__":
    main()
