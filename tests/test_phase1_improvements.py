# coding: utf-8
from __future__ import annotations

from datetime import timezone
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.api import chat
from app.models import Conversation
from app.services.chat_service import ChatService, build_rag_messages
from app.services.openai_service import OpenAIService
from app.services.rag_service import RAGService
from app.utils.time import utc_now
from config import Config, get_config


def test_utc_now_is_timezone_aware():
    now = utc_now()
    assert now.tzinfo is not None
    assert now.tzinfo == timezone.utc


def test_conversation_has_composite_index():
    index_names = {index.name for index in Conversation.__table__.indexes}
    assert "ix_conversations_session_created" in index_names


def test_config_default_llm_model():
    cfg = get_config()
    assert cfg.DEFAULT_LLM_MODEL
    assert cfg.OPENAI_BASE_URL.startswith("http")
    assert cfg.RAG_INDEX_CACHE_SIZE >= 1
    assert cfg.LLM_CONTEXT_TURN_LIMIT >= 1


def test_openai_service_reads_config_base_url_and_model(monkeypatch):
    cfg = Config()
    cfg.OPENAI_API_KEY = "test-key"
    cfg.OPENAI_BASE_URL = "https://example.test/v1/"
    cfg.DEFAULT_LLM_MODEL = "configured-model"
    monkeypatch.setattr("app.services.openai_service.get_config", lambda: cfg)

    service = OpenAIService()
    assert str(service.client.base_url).startswith("https://example.test/v1/")
    assert service.default_model == "configured-model"


def test_build_rag_messages_without_docs():
    messages = build_rag_messages([], "hello", [])
    assert messages == [{"role": "user", "content": "hello"}]


def test_citations_persisted_and_returned_in_history(client, monkeypatch):
    async def fake_chat_completion(**kwargs):
        return "answer with sources"

    monkeypatch.setattr(chat.chat_service.llm_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(
        chat.rag_service,
        "retrieve_context",
        lambda **_: [
            SimpleNamespace(
                page_content="RAG snippet",
                metadata={"filename": "doc.txt"},
            )
        ],
    )
    monkeypatch.setattr(
        chat.rag_service,
        "citations_from_docs",
        lambda docs: [{"id": "1", "filename": "doc.txt", "snippet": "RAG snippet"}],
    )

    send_resp = client.post(
        "/api/chat/message",
        json={"message": "question", "session_id": "cite-session"},
    )
    assert send_resp.status_code == 200
    assert send_resp.json()["citations"][0]["filename"] == "doc.txt"

    history_resp = client.get("/api/chat/history/cite-session")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert len(history) == 1
    assert history[0]["citations"][0]["filename"] == "doc.txt"


def test_stream_persists_citations(client, db_session, monkeypatch):
    async def fake_stream(**kwargs):
        yield "part1"

    monkeypatch.setattr(chat.chat_service.llm_service, "stream_chat_completion", fake_stream)
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(
        chat.rag_service,
        "citations_from_docs",
        lambda _: [{"id": "1", "filename": "s.txt", "snippet": "snip"}],
    )

    response = client.post(
        "/api/chat/message/stream",
        json={"message": "stream cite", "session_id": "stream-cite"},
    )
    assert response.status_code == 200

    row = (
        db_session.query(Conversation)
        .filter(Conversation.session_id == "stream-cite")
        .one()
    )
    assert row.citations_json is not None
    assert "s.txt" in row.citations_json


def test_rag_index_cache_avoids_repeated_disk_load(monkeypatch):
    service = RAGService.__new__(RAGService)
    service.enabled = True
    service.top_k = 2
    service._cache_max_size = 4
    service._store_cache = OrderedDict()
    load_calls = {"count": 0}

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = []

    def fake_load(_session_id):
        load_calls["count"] += 1
        return fake_store

    monkeypatch.setattr(service, "_load_store_from_disk", fake_load)

    service.retrieve_context("cached-session", "query one")
    service.retrieve_context("cached-session", "query two")

    assert load_calls["count"] == 1
    fake_store.similarity_search.assert_called()


def test_rag_cache_cleared_when_index_empty(tmp_path):
    service = RAGService.__new__(RAGService)
    service.enabled = True
    service.index_root = tmp_path / "indexes"
    service.upload_root = tmp_path / "uploads"
    service._cache_max_size = 4
    service._store_cache = {"empty": MagicMock()}
    service.text_splitter = MagicMock()
    service.embeddings = MagicMock()
    (tmp_path / "uploads" / "empty").mkdir(parents=True)

    count = service._rebuild_index_from_saved_files("empty")
    assert count == 0
    assert "empty" not in service._store_cache


def test_chat_service_unifies_prepare_and_save(db_session, monkeypatch):
    cfg = get_config()
    llm = MagicMock()

    async def fake_chat(**kwargs):
        return "async-result"

    llm.chat_completion = fake_chat
    rag = MagicMock()
    rag.retrieve_context.return_value = []
    rag.citations_from_docs.return_value = []

    service = ChatService(cfg, llm, rag)
    monkeypatch.setattr(
        "app.services.chat_service.get_llm_context_messages",
        lambda *args, **kwargs: [{"role": "user", "content": "old"}],
    )

    import asyncio

    result = asyncio.run(
        service.complete_message(db_session, "svc-session", "new message")
    )
    assert result["ai_response"] == "async-result"
    assert result["session_id"] == "svc-session"
    assert result["timestamp"].endswith("Z")
    rag.retrieve_context.assert_called_once()
    saved = (
        db_session.query(Conversation)
        .filter(Conversation.session_id == "svc-session")
        .one()
    )
    assert saved.user_message == "new message"


def test_database_module_uses_config_database_url():
    from app.database import engine

    cfg = get_config()
    assert cfg.DATABASE_URL
    assert str(engine.url).startswith("sqlite")
