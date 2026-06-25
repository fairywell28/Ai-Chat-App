# coding: utf-8
"""Additional session-feature tests for edge cases and isolation."""
from __future__ import annotations

from datetime import datetime, timedelta

from app.api import chat
from app.models import Conversation
from app.services.session_service import list_recent_sessions


def _add_turn(db, session_id: str, user_msg: str, when: datetime):
    db.add(
        Conversation(
            session_id=session_id,
            user_message=user_msg,
            ai_response=f"reply to {user_msg}",
            created_at=when,
            model_used="gpt-4.1-mini",
        )
    )


def test_sessions_default_cap_is_ten(client, db_session):
    base = datetime(2025, 5, 1, 10, 0, 0)
    for idx in range(12):
        _add_turn(db_session, f"sess-{idx:02d}", f"message {idx}", base + timedelta(minutes=idx))
    db_session.commit()

    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert len(sessions) == 10
    assert sessions[0]["session_id"] == "sess-11"
    assert sessions[-1]["session_id"] == "sess-02"


def test_session_title_uses_latest_user_message(db_session):
    base = datetime(2025, 5, 2, 10, 0, 0)
    _add_turn(db_session, "one", "first message", base)
    _add_turn(db_session, "one", "second message wins title", base + timedelta(minutes=5))
    db_session.commit()

    sessions = list_recent_sessions(db_session, limit=10)
    assert sessions[0]["title"] == "second message wins title"
    assert sessions[0]["message_count"] == 2


def test_long_title_truncated(db_session):
    long_text = "A" * 80
    _add_turn(db_session, "long", long_text, datetime(2025, 5, 3, 10, 0, 0))
    db_session.commit()

    sessions = list_recent_sessions(db_session, limit=1)
    assert len(sessions[0]["title"]) <= 48
    assert sessions[0]["title"].endswith("…")


def test_sessions_do_not_leak_context_between_ids(client, db_session, monkeypatch):
    base = datetime(2025, 5, 4, 8, 0, 0)
    _add_turn(db_session, "only-beta", "beta-only question", base)
    db_session.commit()

    captured = {}

    async def fake_chat_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "beta response"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    response = client.post(
        "/api/chat/message",
        json={"message": "new beta turn", "session_id": "only-beta"},
    )
    assert response.status_code == 200

    messages = captured["messages"]
    assert len(messages) == 3
    assert messages[0]["content"] == "beta-only question"
    assert messages[1]["content"] == "reply to beta-only question"
    assert messages[2]["content"] == "new beta turn"


def test_history_limit_parameter(client, db_session):
    base = datetime(2025, 5, 5, 9, 0, 0)
    for idx in range(5):
        _add_turn(db_session, "s-limit", f"turn {idx}", base + timedelta(minutes=idx))
    db_session.commit()

    response = client.get("/api/chat/history/s-limit", params={"limit": 2})
    assert response.status_code == 200
    history = response.json()
    assert len(history) == 2
    assert history[0]["user_message"] == "turn 3"
    assert history[1]["user_message"] == "turn 4"


def test_sessions_limit_query_clamped(client, db_session):
    base = datetime(2025, 5, 6, 9, 0, 0)
    for idx in range(3):
        _add_turn(db_session, f"c{idx}", f"m{idx}", base + timedelta(minutes=idx))
    db_session.commit()

    over = client.get("/api/chat/sessions", params={"limit": 999})
    assert len(over.json()["sessions"]) == 3

    under = client.get("/api/chat/sessions", params={"limit": 0})
    assert len(under.json()["sessions"]) == 1


def test_second_message_moves_session_to_top(client, db_session, monkeypatch):
    base = datetime(2025, 5, 7, 8, 0, 0)
    _add_turn(db_session, "stale", "old", base)
    _add_turn(db_session, "active", "active old", base + timedelta(hours=1))
    db_session.commit()

    async def fake_chat_completion(**_kwargs):
        return "ok"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    client.post(
        "/api/chat/message",
        json={"message": "stale is now newest", "session_id": "stale"},
    )

    sessions = client.get("/api/chat/sessions").json()["sessions"]
    assert sessions[0]["session_id"] == "stale"
    assert sessions[0]["title"] == "stale is now newest"


def test_llm_context_includes_up_to_ten_prior_turns(client, db_session, monkeypatch):
    base = datetime(2025, 5, 8, 8, 0, 0)
    for idx in range(12):
        _add_turn(db_session, "ctx", f"q{idx}", base + timedelta(minutes=idx))
    db_session.commit()

    captured = {}

    async def fake_chat_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return "ok"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(chat.rag_service, "citations_from_docs", lambda _: [])

    client.post(
        "/api/chat/message",
        json={"message": "latest", "session_id": "ctx"},
    )

    # 10 most recent prior turns => 20 history messages + 1 new user message
    assert len(captured["messages"]) == 21
    assert captured["messages"][0]["content"] == "q2"
    assert captured["messages"][-2]["content"] == "reply to q11"
    assert captured["messages"][-1]["content"] == "latest"
