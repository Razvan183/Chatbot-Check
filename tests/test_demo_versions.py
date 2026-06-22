"""Tests for creating the demo chatbot configurations."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import ChatbotVersion
from scripts.create_demo_versions import DEMO_VERSIONS, create_demo_versions


def test_create_demo_versions_is_repeatable_and_restores_settings() -> None:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine)

    first_result = create_demo_versions(session_factory=test_session_factory)

    with test_session_factory() as session:
        baseline = session.scalar(
            select(ChatbotVersion).where(ChatbotVersion.name == "baseline_v1")
        )
        assert baseline is not None
        baseline.top_k = 99
        session.commit()

    second_result = create_demo_versions(session_factory=test_session_factory)

    with test_session_factory() as session:
        versions = list(
            session.scalars(select(ChatbotVersion).order_by(ChatbotVersion.name))
        )

    test_engine.dispose()

    assert first_result == (4, 0)
    assert second_result == (0, 4)
    assert len(versions) == 4
    assert {version.name for version in versions} == {
        version["name"] for version in DEMO_VERSIONS
    }
    assert next(
        version for version in versions if version.name == "baseline_v1"
    ).top_k == 3


def test_demo_versions_match_roadmap_configurations() -> None:
    settings = {
        version["name"]: (
            version["chunk_size"],
            version["chunk_overlap"],
            version["top_k"],
            version["temperature"],
        )
        for version in DEMO_VERSIONS
    }

    assert settings == {
        "baseline_v1": (500, 100, 3, 0.2),
        "more_context_v2": (500, 100, 5, 0.2),
        "strict_refusal_v3": (500, 100, 5, 0.0),
        "weak_bad_demo_v4": (500, 100, 1, 0.7),
    }
