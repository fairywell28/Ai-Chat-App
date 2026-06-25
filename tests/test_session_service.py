# coding: utf-8
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models import Conversation, UserSettings
from app.services.session_service import (
    get_session_history,
    list_recent_sessions,
)
from app.utils.time import serialize_utc_datetime


def test_serialize_utc_datetime_naive_assumes_utc():
    naive = datetime(2025, 6, 16, 11, 13, 34, 748185)
    assert serialize_utc_datetime(naive) == "2025-06-16T11:13:34.748185Z"


def test_serialize_utc_datetime_aware_utc_uses_z_suffix():
    aware = datetime(2025, 6, 16, 11, 13, 34, tzinfo=timezone.utc)
    assert serialize_utc_datetime(aware) == "2025-06-16T11:13:34+00:00".replace("+00:00", "Z")


def test_serialize_utc_datetime_none():
    assert serialize_utc_datetime(None) is None


def test_list_recent_sessions_updated_at_has_utc_suffix(db_session):
    _add_turn(
        db_session,
        "tz-session",
        "hello",
        "world",
        datetime(2025, 6, 16, 11, 0, 0),
    )
    db_session.commit()

    sessions = list_recent_sessions(db_session, limit=1)
    assert sessions[0]["updated_at"].endswith("Z")


def test_get_session_history_created_at_has_utc_suffix(db_session):
    _add_turn(
        db_session,
        "tz-history",
        "hello",
        "world",
        datetime(2025, 6, 16, 11, 0, 0),
    )
    db_session.commit()

    history = get_session_history(db_session, "tz-history")
    assert history[0]["created_at"].endswith("Z")


def _add_turn(db, session_id: str, user_msg: str, ai_msg: str, when: datetime):
    db.add(
        Conversation(
            session_id=session_id,
            user_message=user_msg,
            ai_response=ai_msg,
            created_at=when,
            model_used="gpt-4.1-mini",
        )
    )


def test_list_recent_sessions_orders_by_latest_activity(db_session):
    base = datetime(2025, 1, 1, 12, 0, 0)
    _add_turn(db_session, "older", "old question", "old answer", base)
    _add_turn(
        db_session,
        "newer",
        "latest question here",
        "latest answer",
        base + timedelta(hours=2),
    )
    db_session.commit()

    sessions = list_recent_sessions(db_session, limit=10)
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == "newer"
    assert sessions[1]["session_id"] == "older"
    assert sessions[0]["title"] == "latest question here"
    assert sessions[0]["message_count"] == 1


def test_list_recent_sessions_respects_limit(db_session):
    base = datetime(2025, 2, 1, 10, 0, 0)
    for idx in range(5):
        _add_turn(
            db_session,
            f"s{idx}",
            f"msg {idx}",
            f"reply {idx}",
            base + timedelta(minutes=idx),
        )
    db_session.commit()

    sessions = list_recent_sessions(db_session, limit=3)
    assert len(sessions) == 3
    assert sessions[0]["session_id"] == "s4"


def test_list_recent_sessions_empty(db_session):
    assert list_recent_sessions(db_session) == []


def test_get_session_history_chronological(db_session):
    base = datetime(2025, 3, 1, 9, 0, 0)
    _add_turn(db_session, "s1", "first", "r1", base)
    _add_turn(db_session, "s1", "second", "r2", base + timedelta(minutes=1))
    db_session.commit()

    history = get_session_history(db_session, "s1")
    assert len(history) == 2
    assert history[0]["user_message"] == "first"
    assert history[1]["user_message"] == "second"


def test_get_session_history_unknown_session(db_session):
    assert get_session_history(db_session, "missing") == []


def test_get_llm_context_messages_uses_most_recent_turns(db_session):
    from app.services.session_service import get_llm_context_messages

    base = datetime(2025, 6, 1, 10, 0, 0)
    for idx in range(12):
        _add_turn(
            db_session,
            "ctx",
            f"q{idx}",
            f"r{idx}",
            base + timedelta(minutes=idx),
        )
    db_session.commit()

    messages = get_llm_context_messages(db_session, "ctx", turn_limit=10)
    assert len(messages) == 20
    assert messages[0]["content"] == "q2"
    assert messages[-1]["content"] == "r11"
