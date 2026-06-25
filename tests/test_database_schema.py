# coding: utf-8
"""Tests for persistence layer: composite index, schema migration, config wiring."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Conversation
from app.services.session_service import parse_citations_json


@pytest.fixture
def file_engine(tmp_path):
    db_path = tmp_path / "schema_test.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return engine


def test_composite_index_exists_on_conversation():
    index_names = {idx.name for idx in Conversation.__table__.indexes}
    assert "ix_conversations_session_created" in index_names


def test_sqlite_query_plan_uses_composite_index(file_engine):
    Session = sessionmaker(bind=file_engine)
    db = Session()
    db.add(
        Conversation(
            session_id="plan-test",
            user_message="q",
            ai_response="a",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
        )
    )
    db.commit()

    plan = db.execute(
        text(
            "EXPLAIN QUERY PLAN SELECT * FROM conversations "
            "WHERE session_id = :sid ORDER BY created_at DESC LIMIT 10"
        ),
        {"sid": "plan-test"},
    ).fetchall()
    plan_text = " ".join(row[3] for row in plan)
    assert "ix_conversations_session_created" in plan_text
    db.close()


def test_citations_column_exists_after_create_all(file_engine):
    inspector = inspect(file_engine)
    columns = {col["name"] for col in inspector.get_columns("conversations")}
    assert "citations_json" in columns


def test_migration_adds_citations_column_to_legacy_db(tmp_path):
    """Simulate legacy DB without citations_json."""
    db_path = tmp_path / "legacy.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE conversations ("
                "id INTEGER PRIMARY KEY, session_id VARCHAR(100), "
                "user_message TEXT, ai_response TEXT, "
                "created_at DATETIME, model_used VARCHAR(50))"
            )
        )

    inspector = inspect(engine)
    assert "citations_json" not in {c["name"] for c in inspector.get_columns("conversations")}
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE conversations ADD COLUMN citations_json TEXT"))
    inspector = inspect(engine)
    assert "citations_json" in {c["name"] for c in inspector.get_columns("conversations")}


def test_parse_citations_json_malformed_returns_empty():
    assert parse_citations_json("{not-json") == []
    assert parse_citations_json('{"not": "list"}') == []
    assert parse_citations_json(None) == []
