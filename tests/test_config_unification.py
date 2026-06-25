# coding: utf-8
"""Tests verifying config unification across services."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import Config, get_config


def test_config_exposes_required_keys():
    cfg = get_config()
    for key in (
        "OPENAI_BASE_URL",
        "DEFAULT_LLM_MODEL",
        "DATABASE_URL",
        "RAG_TOP_K",
        "RAG_INDEX_CACHE_SIZE",
        "LLM_CONTEXT_TURN_LIMIT",
    ):
        assert hasattr(cfg, key)
        assert getattr(cfg, key) is not None


def test_openai_service_uses_config_not_hardcoded_url(monkeypatch):
    cfg = Config()
    cfg.OPENAI_API_KEY = "test-key"
    cfg.OPENAI_BASE_URL = "https://custom.api.test/v1/"
    cfg.DEFAULT_LLM_MODEL = "my-model"
    monkeypatch.setattr("app.services.openai_service.get_config", lambda: cfg)

    from app.services.openai_service import OpenAIService

    service = OpenAIService()
    assert "custom.api.test" in str(service.client.base_url)
    assert service.default_model == "my-model"


def test_models_default_llm_matches_config_at_import():
    from app.models import Conversation, UserSettings

    cfg = get_config()
    assert Conversation.model_used.default.arg == cfg.DEFAULT_LLM_MODEL
    assert UserSettings.model_preference.default.arg == cfg.DEFAULT_LLM_MODEL


def test_chat_service_user_settings_uses_config_default(db_session):
    from app.services.chat_service import ChatService

    cfg = get_config()
    service = ChatService(cfg, MagicMock(), MagicMock())
    settings = service.get_or_create_user_settings(db_session, "cfg-unify")
    assert settings.model_preference == cfg.DEFAULT_LLM_MODEL


def test_rag_service_embedding_client_uses_config_base_url(monkeypatch):
    cfg = Config()
    cfg.OPENAI_API_KEY = "key"
    cfg.OPENAI_BASE_URL = "https://embed.test/v1/"
    captured = {}

    class FakeEmb:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.services.rag_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.rag_service.OpenAIEmbeddings", FakeEmb)
    monkeypatch.setattr("app.services.rag_service.FAISS", MagicMock())
    monkeypatch.setattr("app.services.rag_service.RecursiveCharacterTextSplitter", MagicMock())

    from app.services.rag_service import RAGService

    svc = RAGService()
    assert svc.enabled is True
    assert captured["base_url"] == "https://embed.test/v1/"
