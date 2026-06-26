"""Tests for importing the demo evaluation dataset."""

import json
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import EvalCase, EvalDataset
from scripts.import_eval_dataset import import_eval_dataset


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evaluation"


def test_import_eval_dataset_is_repeatable() -> None:
    dataset_path = FIXTURE_DIR / "tiny_eval_set.json"
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    first_dataset_id, first_case_count = import_eval_dataset(
        dataset_path=dataset_path,
        session_factory=test_session_factory,
    )
    second_dataset_id, second_case_count = import_eval_dataset(
        dataset_path=dataset_path,
        session_factory=test_session_factory,
    )

    with test_session_factory() as session:
        dataset_count = session.scalar(select(func.count()).select_from(EvalDataset))
        case_count = session.scalar(select(func.count()).select_from(EvalCase))
        first_case = session.scalar(select(EvalCase).order_by(EvalCase.id))

    test_engine.dispose()

    assert first_case_count == second_case_count == 2
    assert first_dataset_id > 0
    assert second_dataset_id > 0
    assert dataset_count == 1
    assert case_count == 2
    assert json.loads(first_case.expected_chunk_ids) == [1, 2]
    assert json.loads(first_case.expected_chunk_keys) == ["vacation_policy.md::0"]
