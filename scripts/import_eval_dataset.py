"""Import the demo evaluation dataset into the database."""

from pathlib import Path
import json
import sys
from collections.abc import Callable


# Allow both module execution and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, create_database_tables
from app.db.models import EvalCase, EvalDataset, EvalResult, EvalRun


DEFAULT_DATASET_PATH = Path("data") / "evaluation" / "hr_policy_eval_set.json"


def _load_dataset_file(dataset_path: str | Path) -> dict:
    """Read and validate the eval dataset JSON file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {"name", "description", "domain", "cases"}
    missing_fields = required_fields - set(data)
    if missing_fields:
        raise ValueError(
            "Evaluation dataset is missing fields: "
            + ", ".join(sorted(missing_fields))
        )
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("Evaluation dataset must include at least one case")

    return data


def _clear_existing_dataset(database_session: Session, dataset_name: str) -> None:
    """Remove an existing dataset and dependent eval rows by name."""
    dataset = database_session.scalar(
        select(EvalDataset).where(EvalDataset.name == dataset_name)
    )
    if dataset is None:
        return

    case_ids = list(
        database_session.scalars(
            select(EvalCase.id).where(EvalCase.eval_dataset_id == dataset.id)
        )
    )
    run_ids = list(
        database_session.scalars(
            select(EvalRun.id).where(EvalRun.eval_dataset_id == dataset.id)
        )
    )

    if run_ids:
        database_session.execute(
            delete(EvalResult).where(EvalResult.eval_run_id.in_(run_ids))
        )
        database_session.execute(delete(EvalRun).where(EvalRun.id.in_(run_ids)))
    if case_ids:
        database_session.execute(
            delete(EvalResult).where(EvalResult.eval_case_id.in_(case_ids))
        )
        database_session.execute(delete(EvalCase).where(EvalCase.id.in_(case_ids)))

    database_session.delete(dataset)


def import_eval_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    session_factory: Callable[[], Session] = SessionLocal,
) -> tuple[int, int]:
    """Replace the demo eval dataset and return the dataset id and case count."""
    data = _load_dataset_file(dataset_path)
    create_database_tables()

    with session_factory() as database_session:
        try:
            _clear_existing_dataset(database_session, data["name"])
            database_session.flush()

            dataset = EvalDataset(
                name=data["name"],
                description=data["description"],
                domain=data["domain"],
            )
            database_session.add(dataset)
            database_session.flush()

            for case_data in data["cases"]:
                database_session.add(
                    EvalCase(
                        eval_dataset_id=dataset.id,
                        question=case_data["question"],
                        expected_answer=case_data.get("expected_answer"),
                        expected_chunk_ids=json.dumps(
                            case_data.get("expected_chunk_ids", [])
                        ),
                        expected_chunk_keys=json.dumps(
                            case_data.get("expected_chunk_keys", [])
                        ),
                        question_type=case_data["question_type"],
                        difficulty=case_data["difficulty"],
                        should_be_answerable=case_data["should_be_answerable"],
                    )
                )

            database_session.commit()
            return dataset.id, len(data["cases"])
        except Exception:
            database_session.rollback()
            raise


def main() -> None:
    """Import the demo eval dataset and print a short summary."""
    dataset_id, case_count = import_eval_dataset()
    print(f"Imported evaluation dataset {dataset_id}")
    print(f"Imported {case_count} evaluation cases")


if __name__ == "__main__":
    main()
