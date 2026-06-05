# coding: utf-8
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Conversation

DEFAULT_SESSION_LIMIT = 10
TITLE_MAX_LEN = 48
PREVIEW_MAX_LEN = 80


def _truncate(text: str, max_len: int) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= max_len:
        return cleaned or "新对话"
    return cleaned[: max_len - 1] + "…"


def conversation_to_history_item(conv: Conversation) -> Dict[str, Any]:
    return {
        "id": conv.id,
        "user_message": conv.user_message,
        "ai_response": conv.ai_response,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "model_used": conv.model_used,
    }


def get_llm_context_messages(
    db: Session,
    session_id: str,
    *,
    turn_limit: int = 10,
) -> List[Dict[str, str]]:
    """Last N turns formatted for the chat completion API (chronological)."""
    rows = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .limit(turn_limit)
        .all()
    )
    messages: List[Dict[str, str]] = []
    for conv in reversed(rows):
        messages.append({"role": "user", "content": conv.user_message})
        messages.append({"role": "assistant", "content": conv.ai_response})
    return messages


def get_session_history(
    db: Session,
    session_id: str,
    *,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return chronological turn history for one session."""
    rows = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .all()
    )
    return [conversation_to_history_item(conv) for conv in reversed(rows)]


def list_recent_sessions(
    db: Session,
    *,
    limit: int = DEFAULT_SESSION_LIMIT,
) -> List[Dict[str, Any]]:
    """List sessions ordered by most recent activity."""
    aggregates = (
        db.query(
            Conversation.session_id.label("session_id"),
            func.max(Conversation.created_at).label("updated_at"),
            func.count(Conversation.id).label("message_count"),
        )
        .group_by(Conversation.session_id)
        .order_by(func.max(Conversation.created_at).desc())
        .limit(limit)
        .all()
    )

    sessions: List[Dict[str, Any]] = []
    for row in aggregates:
        latest = (
            db.query(Conversation)
            .filter(Conversation.session_id == row.session_id)
            .order_by(Conversation.created_at.desc())
            .first()
        )
        preview_source = latest.user_message if latest else ""
        sessions.append(
            {
                "session_id": row.session_id,
                "title": _truncate(preview_source, TITLE_MAX_LEN),
                "preview": _truncate(preview_source, PREVIEW_MAX_LEN),
                "updated_at": _serialize_dt(row.updated_at),
                "message_count": int(row.message_count or 0),
            }
        )
    return sessions


def _serialize_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
