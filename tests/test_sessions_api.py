# coding: utf-8
from __future__ import annotations

from datetime import datetime, timedelta

from app.api import chat
from app.models import Conversation


def _seed_sessions(db):
    base = datetime(2025, 4, 1, 8, 0, 0)
    db.add(
        Conversation(
            session_id="alpha",
            user_message="Alpha first message",
            ai_response="Alpha reply",
            created_at=base,
            model_used="gpt-4.1-mini",
        )
    )
    db.add(
        Conversation(
            session_id="beta",
            user_message="Beta is newer",
            ai_response="Beta reply",
            created_at=base + timedelta(hours=1),
            model_used="gpt-4.1-mini",
        )
    )
    db.commit()


def test_list_sessions_endpoint(client, db_session):
    _seed_sessions(db_session)
    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    payload = response.json()
    sessions = payload["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == "beta"
    assert sessions[0]["title"] == "Beta is newer"
    assert sessions[1]["session_id"] == "alpha"
    assert sessions[0]["updated_at"].endswith("Z")


def test_list_sessions_limit_query(client, db_session):
    _seed_sessions(db_session)
    response = client.get("/api/chat/sessions", params={"limit": 1})
    assert response.status_code == 200
    assert len(response.json()["sessions"]) == 1


def test_switch_session_loads_history(client, db_session, monkeypatch):
    _seed_sessions(db_session)

    async def fake_chat_completion(**kwargs):
        assert kwargs["messages"][0]["content"] == "Alpha first message"
        assert kwargs["messages"][1]["content"] == "Alpha reply"
        assert kwargs["messages"][2]["role"] == "user"
        assert kwargs["messages"][2]["content"] == "Continue alpha"
        return "continued"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    history_resp = client.get("/api/chat/history/alpha")
    assert history_resp.status_code == 200
    assert len(history_resp.json()) == 1
    assert history_resp.json()[0]["user_message"] == "Alpha first message"

    msg_resp = client.post(
        "/api/chat/message",
        json={"message": "Continue alpha", "session_id": "alpha"},
    )
    assert msg_resp.status_code == 200
    assert msg_resp.json()["session_id"] == "alpha"
    assert msg_resp.json()["ai_response"] == "continued"


def test_new_message_updates_session_list(client, monkeypatch):
    async def fake_chat_completion(**kwargs):
        return "ok"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    send_resp = client.post(
        "/api/chat/message",
        json={"message": "Hello sessions", "session_id": "fresh-session"},
    )
    assert send_resp.status_code == 200

    list_resp = client.get("/api/chat/sessions")
    sessions = list_resp.json()["sessions"]
    assert sessions[0]["session_id"] == "fresh-session"
    assert sessions[0]["title"] == "Hello sessions"
