# coding: utf-8
"""Thorough unit tests for ChatService (Phase 1 dedupe + citations)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models import Conversation, UserSettings
from app.services.chat_service import ChatService, PreparedChat, build_rag_messages
from config import get_config


@pytest.fixture
def chat_service(db_session, monkeypatch):
    cfg = get_config()
    llm = MagicMock()
    rag = MagicMock()
    rag.retrieve_context.return_value = []
    rag.citations_from_docs.return_value = []
    service = ChatService(cfg, llm, rag)
    return service, llm, rag, cfg


def test_build_rag_messages_injects_system_prompt():
    doc = SimpleNamespace(page_content="ctx", metadata={"filename": "a.txt"})
    messages = build_rag_messages([], "question", [doc])
    assert messages[0]["role"] == "system"
    assert "a.txt" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "question"}


def test_resolve_session_id_generates_uuid_when_missing(chat_service):
    service, *_ = chat_service
    sid = service.resolve_session_id(None)
    assert sid
    assert service.resolve_session_id("existing") == "existing"


def test_get_or_create_user_settings_uses_config_default_model(chat_service, db_session):
    service, _, _, cfg = chat_service
    settings = service.get_or_create_user_settings(db_session, "settings-session")
    assert settings.model_preference == cfg.DEFAULT_LLM_MODEL


def test_get_or_create_user_settings_idempotent(chat_service, db_session):
    service, _, _, _ = chat_service
    first = service.get_or_create_user_settings(db_session, "dup-settings")
    second = service.get_or_create_user_settings(db_session, "dup-settings")
    assert first.id == second.id
    assert db_session.query(UserSettings).count() == 1


def test_prepare_chat_passes_rag_top_k_from_config(chat_service, db_session):
    service, _, rag, cfg = chat_service
    service.prepare_chat(db_session, "prep-session", "hello")
    rag.retrieve_context.assert_called_once_with(
        session_id="prep-session",
        query="hello",
        k=cfg.RAG_TOP_K,
    )


def test_save_conversation_turn_persists_citations_json(chat_service, db_session):
    service, _, _, cfg = chat_service
    settings = service.get_or_create_user_settings(db_session, "save-cite")
    prepared = PreparedChat(
        session_id="save-cite",
        user_message="q",
        user_settings=settings,
        llm_messages=[],
        citations=[{"id": "1", "filename": "f.txt", "snippet": "s"}],
    )
    service.save_conversation_turn(db_session, prepared, "answer")
    row = db_session.query(Conversation).one()
    assert json.loads(row.citations_json)[0]["filename"] == "f.txt"


def test_save_conversation_turn_null_citations_when_empty(chat_service, db_session):
    service, _, _, _ = chat_service
    settings = service.get_or_create_user_settings(db_session, "no-cite")
    prepared = PreparedChat(
        session_id="no-cite",
        user_message="q",
        user_settings=settings,
        llm_messages=[],
        citations=[],
    )
    service.save_conversation_turn(db_session, prepared, "answer")
    row = db_session.query(Conversation).one()
    assert row.citations_json is None


def test_complete_message_timestamp_ends_with_z(chat_service, db_session):
    service, llm, _, _ = chat_service

    async def fake_chat(**kwargs):
        return "ok"

    llm.chat_completion = fake_chat
    result = asyncio.run(service.complete_message(db_session, "ts-session", "hi"))
    assert result["timestamp"].endswith("Z")


def test_stream_and_complete_share_prepare_logic(chat_service, db_session, monkeypatch):
    """Stream and non-stream must use the same prepare_chat path."""
    service, llm, rag, _ = chat_service
    prepare_calls = {"count": 0}
    original_prepare = service.prepare_chat

    def counting_prepare(*args, **kwargs):
        prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(service, "prepare_chat", counting_prepare)

    async def fake_chat(**kwargs):
        return "sync"

    async def fake_stream(**kwargs):
        yield "chunk"

    llm.chat_completion = fake_chat
    llm.stream_chat_completion = fake_stream

    asyncio.run(service.complete_message(db_session, "parity", "msg1"))

    async def consume_stream():
        events = []
        async for event in service.stream_message(db_session, "parity", "msg2"):
            events.append(event)
        return events

    events = asyncio.run(consume_stream())
    assert prepare_calls["count"] == 2
    assert json.loads(events[0])["citations"] == []
    assert db_session.query(Conversation).filter(Conversation.session_id == "parity").count() == 2


def test_stream_saves_full_concatenated_response(chat_service, db_session):
    service, llm, _, _ = chat_service

    async def fake_stream(**kwargs):
        yield "Hello"
        yield " "
        yield "World"

    llm.stream_chat_completion = fake_stream

    async def consume():
        async for _ in service.stream_message(db_session, "stream-save", "q"):
            pass

    asyncio.run(consume())
    row = db_session.query(Conversation).filter(Conversation.session_id == "stream-save").one()
    assert row.ai_response == "Hello World"
