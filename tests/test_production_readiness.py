# coding: utf-8
"""Production-readiness tests: error paths, API contracts, persistence edge cases."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.api import chat
from app.database import _ensure_conversations_citations_column
from app.models import Base, Conversation, UserSettings
from app.services.chat_service import ChatService
from app.utils.time import utc_now
from config import get_config


def _stub_llm(monkeypatch, reply: str = "ok"):
    async def fake_chat(**kwargs):
        return reply

    async def fake_stream(**kwargs):
        yield reply

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat)
    monkeypatch.setattr(chat.chat_service.llm_service, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])


# --- API contract & validation ---


def test_message_requires_message_field(client):
    response = client.post("/api/chat/message", json={"session_id": "s1"})
    assert response.status_code == 422


def test_stream_requires_message_field(client):
    response = client.post("/api/chat/message/stream", json={})
    assert response.status_code == 422


def test_history_unknown_session_returns_empty_list(client):
    response = client.get("/api/chat/history/nonexistent-session-id")
    assert response.status_code == 200
    assert response.json() == []


def test_sessions_empty_database_returns_empty_list(client):
    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    assert response.json()["sessions"] == []


def test_sessions_limit_zero_clamped_to_one(client, db_session):
    db_session.add(
        Conversation(
            session_id="clamp-test",
            user_message="hello",
            ai_response="hi",
            created_at=utc_now(),
        )
    )
    db_session.commit()
    response = client.get("/api/chat/sessions", params={"limit": 0})
    assert response.status_code == 200
    assert len(response.json()["sessions"]) == 1


def test_auto_generated_session_ids_are_unique(client, monkeypatch):
    _stub_llm(monkeypatch)
    r1 = client.post("/api/chat/message", json={"message": "a"})
    r2 = client.post("/api/chat/message", json={"message": "b"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["session_id"] != r2.json()["session_id"]


# --- Error handling ---


def test_sync_message_llm_failure_returns_500(client, monkeypatch):
    async def failing_chat(**kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", failing_chat)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])

    response = client.post(
        "/api/chat/message",
        json={"message": "fail", "session_id": "fail-sync"},
    )
    assert response.status_code == 500
    assert "LLM unavailable" in response.json()["detail"]


def test_sync_message_llm_failure_does_not_persist_row(client, db_session, monkeypatch):
    async def failing_chat(**kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", failing_chat)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])

    client.post(
        "/api/chat/message",
        json={"message": "fail", "session_id": "no-persist"},
    )
    count = (
        db_session.query(Conversation)
        .filter(Conversation.session_id == "no-persist")
        .count()
    )
    assert count == 0


def test_stream_llm_failure_emits_error_and_skips_save(client, db_session, monkeypatch):
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    async def failing_stream(**kwargs):
        if False:
            yield ""
        raise RuntimeError("stream broke")

    monkeypatch.setattr(
        chat.chat_service.llm_service,
        "stream_chat_completion",
        failing_stream,
    )

    response = client.post(
        "/api/chat/message/stream",
        json={"message": "stream fail", "session_id": "stream-fail"},
    )
    assert response.status_code == 200
    assert "stream broke" in response.text
    assert (
        db_session.query(Conversation)
        .filter(Conversation.session_id == "stream-fail")
        .count()
        == 0
    )


def test_upload_internal_error_returns_500(client, monkeypatch):
    async def boom(_session_id, _files):
        raise ValueError("disk full")

    monkeypatch.setattr(chat.rag_service, "ingest_files", boom)
    response = client.post(
        "/api/chat/upload",
        data={"session_id": "s1"},
        files=[("files", ("a.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 500
    assert "disk full" in response.json()["detail"]


def test_reindex_failure_returns_500(client, monkeypatch):
    def boom(_session_id):
        raise RuntimeError("reindex failed")

    monkeypatch.setattr(chat.rag_service, "reindex_session", boom)
    response = client.post("/api/chat/files/s1/reindex")
    assert response.status_code == 500
    assert "reindex failed" in response.json()["detail"]


def test_reindex_returns_503_when_rag_disabled(client, monkeypatch):
    monkeypatch.setattr(chat.rag_service, "enabled", False, raising=False)
    monkeypatch.setattr(chat.rag_service, "disabled_reason", "disabled", raising=False)
    response = client.post("/api/chat/files/s1/reindex")
    assert response.status_code == 503


def test_list_files_returns_503_when_rag_disabled(client, monkeypatch):
    monkeypatch.setattr(chat.rag_service, "enabled", False, raising=False)
    monkeypatch.setattr(chat.rag_service, "disabled_reason", "disabled", raising=False)
    response = client.get("/api/chat/files/s1")
    assert response.status_code == 503


# --- Data integrity ---


def test_message_turn_order_preserved_in_history(client, monkeypatch):
    _stub_llm(monkeypatch, reply="reply")
    session_id = "order-test"
    for i in range(3):
        client.post(
            "/api/chat/message",
            json={"message": f"msg-{i}", "session_id": session_id},
        )
    history = client.get(f"/api/chat/history/{session_id}").json()
    assert [h["user_message"] for h in history] == ["msg-0", "msg-1", "msg-2"]


def test_history_malformed_citations_returned_as_empty(client, db_session):
    db_session.add(
        Conversation(
            session_id="bad-json",
            user_message="q",
            ai_response="a",
            citations_json="{not valid json",
            created_at=utc_now(),
        )
    )
    db_session.commit()
    history = client.get("/api/chat/history/bad-json").json()
    assert history[0]["citations"] == []


def test_user_settings_created_once_per_session(client, db_session, monkeypatch):
    _stub_llm(monkeypatch)
    session_id = "settings-once"
    client.post("/api/chat/message", json={"message": "one", "session_id": session_id})
    client.post("/api/chat/message", json={"message": "two", "session_id": session_id})
    count = (
        db_session.query(UserSettings)
        .filter(UserSettings.session_id == session_id)
        .count()
    )
    assert count == 1


def test_llm_receives_temperature_and_max_tokens_from_settings(
    client, db_session, monkeypatch
):
    captured = {}

    async def capture_chat(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", capture_chat)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])

    db_session.add(
        UserSettings(
            session_id="settings-params",
            temperature=5,
            max_tokens=512,
            model_preference=get_config().DEFAULT_LLM_MODEL,
        )
    )
    db_session.commit()

    client.post(
        "/api/chat/message",
        json={"message": "params", "session_id": "settings-params"},
    )
    assert captured["temperature"] == pytest.approx(0.5)
    assert captured["max_tokens"] == 512


# --- Database migration ---


def test_ensure_citations_column_adds_to_legacy_table(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_prod.db"
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

    import app.database as db_module

    monkeypatch.setattr(db_module, "engine", engine)
    _ensure_conversations_citations_column()

    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("conversations")}
    assert "citations_json" in columns

    _ensure_conversations_citations_column()
    assert len(inspector.get_columns("conversations")) == len(columns)


def test_create_tables_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    assert "conversations" in inspector.get_table_names()
    assert "user_settings" in inspector.get_table_names()


# --- ChatService unit edge cases ---


def test_prepare_chat_with_eleven_prior_turns_limits_context(db_session):
    cfg = get_config()
    service = ChatService(cfg, MagicMock(), MagicMock())
    session_id = "context-limit"
    for i in range(11):
        db_session.add(
            Conversation(
                session_id=session_id,
                user_message=f"u{i}",
                ai_response=f"a{i}",
                created_at=datetime(2025, 1, 1, 0, 0, i),
            )
        )
    db_session.commit()

    prepared = service.prepare_chat(db_session, session_id, "new question")
    user_contents = [
        m["content"] for m in prepared.llm_messages if m["role"] == "user"
    ]
    assert "u1" in user_contents
    assert "u0" not in user_contents
    assert user_contents[-1] == "new question"


def test_stream_yields_valid_json_events(chat_service, db_session):
    service, llm, _, _ = chat_service

    async def fake_stream(**kwargs):
        yield "part1"
        yield "part2"

    llm.stream_chat_completion = fake_stream

    async def collect():
        events = []
        async for raw in service.stream_message(db_session, "json-events", "q"):
            parsed = json.loads(raw)
            events.append(parsed)
        return events

    import asyncio

    events = asyncio.run(collect())
    assert "citations" in events[0]
    assert events[1]["content"] == "part1"
    assert events[2]["content"] == "part2"


@pytest.fixture
def chat_service(db_session, monkeypatch):
    cfg = get_config()
    llm = MagicMock()
    rag = MagicMock()
    rag.retrieve_context.return_value = []
    rag.citations_from_docs.return_value = []
    service = ChatService(cfg, llm, rag)
    return service, llm, rag, cfg
