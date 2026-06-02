# coding: utf-8
from __future__ import annotations

from types import SimpleNamespace

from app.api import chat


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_message_creates_session_and_returns_citations(client, monkeypatch):
    async def fake_chat_completion(**kwargs):
        assert kwargs["model"] == "gpt-4.1-mini"
        return "这是测试回复"

    monkeypatch.setattr(chat.openai_service, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(
        chat.rag_service,
        "retrieve_context",
        lambda **_: [
            SimpleNamespace(
                page_content="RAG 内容片段",
                metadata={"filename": "manual.txt"},
            )
        ],
    )
    monkeypatch.setattr(
        chat.rag_service,
        "citations_from_docs",
        lambda docs: [{"id": "1", "filename": "manual.txt", "snippet": "RAG 内容片段"}],
    )

    response = client.post("/api/chat/message", json={"message": "你好"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["ai_response"] == "这是测试回复"
    assert payload["citations"][0]["filename"] == "manual.txt"


def test_upload_returns_503_when_rag_disabled(client, monkeypatch):
    monkeypatch.setattr(chat.rag_service, "enabled", False, raising=False)
    monkeypatch.setattr(chat.rag_service, "disabled_reason", "RAG disabled for test", raising=False)

    response = client.post(
        "/api/chat/upload",
        data={"session_id": "s1"},
        files=[("files", ("a.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 503
    assert "RAG disabled for test" in response.text


def test_upload_rejects_unsupported_files(client, monkeypatch):
    async def fake_ingest(_session_id, _files):
        return 0, []

    monkeypatch.setattr(chat.rag_service, "ingest_files", fake_ingest)
    response = client.post(
        "/api/chat/upload",
        data={"session_id": "s1"},
        files=[("files", ("image.png", b"not text", "image/png"))],
    )
    assert response.status_code == 400
    assert "可支持的文件类型" in response.json()["detail"]


def test_file_management_endpoints(client, monkeypatch):
    monkeypatch.setattr(
        chat.rag_service,
        "list_files",
        lambda session_id: [{"filename": "doc1.txt", "saved_filename": "uuid_doc1.txt", "size_bytes": "10"}],
    )
    monkeypatch.setattr(chat.rag_service, "delete_file", lambda session_id, filename: filename == "doc1.txt")
    monkeypatch.setattr(chat.rag_service, "reindex_session", lambda session_id: 12)

    list_resp = client.get("/api/chat/files/s1")
    assert list_resp.status_code == 200
    assert list_resp.json()["files"][0]["filename"] == "doc1.txt"

    delete_resp = client.delete("/api/chat/files/s1", params={"filename": "doc1.txt"})
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted_file"] == "doc1.txt"

    missing_resp = client.delete("/api/chat/files/s1", params={"filename": "missing.txt"})
    assert missing_resp.status_code == 404

    reindex_resp = client.post("/api/chat/files/s1/reindex")
    assert reindex_resp.status_code == 200
    assert reindex_resp.json()["indexed_chunks"] == 12


def test_stream_emits_citations_first(client, monkeypatch):
    monkeypatch.setattr(chat.rag_service, "retrieve_context", lambda **_: [])
    monkeypatch.setattr(
        chat.rag_service,
        "citations_from_docs",
        lambda _: [{"id": "1", "filename": "doc.txt", "snippet": "snippet"}],
    )

    def fake_stream_chat_completion(**kwargs):
        async def iterator():
            yield "A"
            yield "B"
        return iterator()

    monkeypatch.setattr(chat.openai_service, "stream_chat_completion", fake_stream_chat_completion)

    response = client.post(
        "/api/chat/message/stream",
        json={"message": "stream test", "session_id": "s-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert '"citations"' in body
    assert '"content": "A"' in body or '"content":"A"' in body
