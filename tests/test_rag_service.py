# coding: utf-8
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.services.rag_service import RAGService


def test_citations_from_docs_truncates_long_snippet():
    service = RAGService.__new__(RAGService)
    docs = [
        SimpleNamespace(
            page_content="x" * 250,
            metadata={"filename": "book.txt"},
        )
    ]
    citations = service.citations_from_docs(docs)
    assert citations[0]["filename"] == "book.txt"
    assert citations[0]["snippet"].endswith("...")
    assert len(citations[0]["snippet"]) <= 183


def test_display_name_strips_uuid_prefix():
    path = Path("abc123_file.md")
    assert RAGService._display_name(path) == "file.md"


def test_retrieve_context_returns_empty_when_disabled():
    service = RAGService.__new__(RAGService)
    service.enabled = False
    assert service.retrieve_context("s1", "hello") == []
