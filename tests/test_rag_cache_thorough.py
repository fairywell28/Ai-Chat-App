# coding: utf-8
"""Thorough unit tests for RAG in-memory index cache (Phase 1)."""
from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from app.services.rag_service import RAGService


def _make_service(cache_size: int = 2) -> RAGService:
    service = RAGService.__new__(RAGService)
    service.enabled = True
    service.top_k = 3
    service._cache_max_size = cache_size
    service._store_cache = OrderedDict()
    return service


def test_lru_evicts_oldest_when_cache_full(monkeypatch):
    service = _make_service(cache_size=2)
    stores = {
        "s1": MagicMock(name="s1"),
        "s2": MagicMock(name="s2"),
        "s3": MagicMock(name="s3"),
    }

    def fake_load(session_id):
        return stores[session_id]

    monkeypatch.setattr(service, "_load_store_from_disk", fake_load)

    service.retrieve_context("s1", "q")
    service.retrieve_context("s2", "q")
    service.retrieve_context("s3", "q")  # should evict s1

    assert list(service._store_cache.keys()) == ["s2", "s3"]
    assert "s1" not in service._store_cache


def test_cache_hit_moves_entry_to_end(monkeypatch):
    service = _make_service(cache_size=2)
    stores = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}

    monkeypatch.setattr(service, "_load_store_from_disk", lambda sid: stores[sid])

    service.retrieve_context("a", "q")
    service.retrieve_context("b", "q")
    service.retrieve_context("a", "q")  # refresh a to MRU
    service.retrieve_context("c", "q")  # evict b

    assert "b" not in service._store_cache
    assert list(service._store_cache.keys()) == ["a", "c"]


def test_save_store_updates_cache_without_disk_load(monkeypatch):
    service = _make_service(cache_size=4)
    store = MagicMock()
    store.similarity_search.return_value = ["chunk"]
    load_calls = {"count": 0}

    def fake_load(_sid):
        load_calls["count"] += 1
        return MagicMock()

    monkeypatch.setattr(service, "_load_store_from_disk", fake_load)
    monkeypatch.setattr(service, "_session_index_path", lambda _sid: MagicMock(mkdir=MagicMock()))

    service._save_store("saved", store)
    assert service._store_cache["saved"] is store

    result = service.retrieve_context("saved", "q")
    assert load_calls["count"] == 0
    assert result == ["chunk"]
    store.similarity_search.assert_called_once()


def test_invalidate_removes_session_from_cache():
    service = _make_service()
    service._store_cache["gone"] = MagicMock()
    service._invalidate_store_cache("gone")
    assert "gone" not in service._store_cache


def test_clear_index_cache_empties_all():
    service = _make_service()
    service._store_cache["a"] = MagicMock()
    service.clear_index_cache()
    assert len(service._store_cache) == 0
