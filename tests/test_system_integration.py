# coding: utf-8
"""System-level integration tests across chat, sessions, RAG, and persistence."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.api import chat
from app.models import Conversation, UserSettings
from config import get_config


def _fake_llm(monkeypatch):
    async def fake_chat(**kwargs):
        return f"reply to: {kwargs['messages'][-1]['content']}"

    async def fake_stream(**kwargs):
        yield "stream-"
        yield "reply"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat)
    monkeypatch.setattr(chat.chat_service.llm_service, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])


def test_full_session_lifecycle(client, db_session, monkeypatch):
    """New session → message → history → sessions list → second message with context."""
    _fake_llm(monkeypatch)
    cfg = get_config()

    r1 = client.post(
        "/api/chat/message",
        json={"message": "first question", "session_id": "lifecycle-1"},
    )
    assert r1.status_code == 200
    assert r1.json()["timestamp"].endswith("Z")

    history = client.get("/api/chat/history/lifecycle-1").json()
    assert len(history) == 1
    assert history[0]["created_at"].endswith("Z")

    sessions = client.get("/api/chat/sessions").json()["sessions"]
    assert sessions[0]["session_id"] == "lifecycle-1"
    assert sessions[0]["updated_at"].endswith("Z")

    captured = {}

    async def capture_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "second reply"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", capture_chat)

    r2 = client.post(
        "/api/chat/message",
        json={"message": "follow up", "session_id": "lifecycle-1"},
    )
    assert r2.status_code == 200
    assert len(captured["messages"]) == 3
    assert captured["messages"][0]["content"] == "first question"
    assert captured["messages"][1]["content"].startswith("reply to:")
    assert captured["messages"][2]["content"] == "follow up"

    settings = (
        db_session.query(UserSettings)
        .filter(UserSettings.session_id == "lifecycle-1")
        .one()
    )
    assert settings.model_preference == cfg.DEFAULT_LLM_MODEL


def test_stream_and_sync_persist_equivalent_rows(client, db_session, monkeypatch):
    _fake_llm(monkeypatch)

    sync = client.post(
        "/api/chat/message",
        json={"message": "sync msg", "session_id": "sync-s"},
    )
    assert sync.status_code == 200

    stream = client.post(
        "/api/chat/message/stream",
        json={"message": "stream msg", "session_id": "stream-s"},
    )
    assert stream.status_code == 200

    sync_row = db_session.query(Conversation).filter(Conversation.session_id == "sync-s").one()
    stream_row = db_session.query(Conversation).filter(Conversation.session_id == "stream-s").one()
    assert sync_row.user_message == "sync msg"
    assert stream_row.user_message == "stream msg"
    assert stream_row.ai_response == "stream-reply"


def test_citations_roundtrip_through_history(client, monkeypatch):
    async def fake_chat(**kwargs):
        return "with cites"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat)
    monkeypatch.setattr(
        chat.rag_service,
        "retrieve_context",
        lambda **_: [SimpleNamespace(page_content="doc body", metadata={"filename": "ref.pdf"})],
    )
    monkeypatch.setattr(
        chat.rag_service,
        "citations_from_docs",
        lambda _: [{"id": "1", "filename": "ref.pdf", "snippet": "doc body"}],
    )

    client.post(
        "/api/chat/message",
        json={"message": "cite test", "session_id": "cite-roundtrip"},
    )
    history = client.get("/api/chat/history/cite-roundtrip").json()
    assert history[0]["citations"][0]["filename"] == "ref.pdf"


def test_sessions_isolated_by_session_id(client, db_session, monkeypatch):
    _fake_llm(monkeypatch)

    client.post("/api/chat/message", json={"message": "alpha", "session_id": "iso-alpha"})
    client.post("/api/chat/message", json={"message": "beta", "session_id": "iso-beta"})

    alpha_history = client.get("/api/chat/history/iso-alpha").json()
    beta_history = client.get("/api/chat/history/iso-beta").json()
    assert len(alpha_history) == 1
    assert len(beta_history) == 1
    assert alpha_history[0]["user_message"] == "alpha"
    assert beta_history[0]["user_message"] == "beta"


def test_message_timestamp_uses_utc_z_suffix(client, monkeypatch):
    _fake_llm(monkeypatch)
    resp = client.post("/api/chat/message", json={"message": "time check", "session_id": "tz-check"})
    assert resp.json()["timestamp"].endswith("Z")


def test_root_and_health(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
