# coding: utf-8
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.api import chat
from app.models import Base, Conversation, UserSettings


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def db_session(test_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[chat.get_db] = override_get_db
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clean_db_tables(db_session):
    db_session.query(Conversation).delete()
    db_session.query(UserSettings).delete()
    db_session.commit()
    yield


@pytest.fixture(autouse=True)
def reset_rag_flags(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(chat.rag_service, "enabled", True, raising=False)
    monkeypatch.setattr(chat.rag_service, "disabled_reason", "", raising=False)
    monkeypatch.setattr(chat.rag_service, "top_k", 3, raising=False)
    monkeypatch.setattr(chat.chat_service.cfg, "RAG_TOP_K", 3, raising=False)
    chat.rag_service.clear_index_cache()
