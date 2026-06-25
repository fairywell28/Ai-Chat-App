# coding: utf-8
from __future__ import annotations

import shutil
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict
from uuid import uuid4

from fastapi import UploadFile
from config import get_config

try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency fallback
    Document = Any
    RecursiveCharacterTextSplitter = None
    FAISS = None
    OpenAIEmbeddings = None
    PdfReader = None


logger = logging.getLogger(__name__)


class RAGService:
    """Manage per-session document ingestion and retrieval for RAG."""

    ALLOWED_SUFFIXES = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".pdf",
        ".py",
        ".log",
    }

    def __init__(self) -> None:
        cfg = get_config()
        self.index_root = Path("data/rag_indexes")
        self.upload_root = Path("data/uploads")
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        self.disabled_reason = ""
        self.top_k = cfg.RAG_TOP_K
        self._cache_max_size = max(1, cfg.RAG_INDEX_CACHE_SIZE)
        self._store_cache: OrderedDict[str, Any] = OrderedDict()

        api_key = cfg.OPENAI_API_KEY
        base_url = cfg.OPENAI_BASE_URL
        if OpenAIEmbeddings is None or FAISS is None or RecursiveCharacterTextSplitter is None:
            self.enabled = False
            self.disabled_reason = "LangChain 相关依赖未安装，RAG不可用。"
            logger.warning(self.disabled_reason)
            return
        if not api_key:
            self.enabled = False
            self.disabled_reason = "OPENAI_API_KEY环境变量未设置，RAG不可用。"
            logger.warning(self.disabled_reason)
            return

        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=cfg.RAG_EMBEDDING_MODEL,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.RAG_CHUNK_SIZE,
            chunk_overlap=cfg.RAG_CHUNK_OVERLAP,
        )

    def clear_index_cache(self) -> None:
        self._store_cache.clear()

    def _session_index_path(self, session_id: str) -> Path:
        return self.index_root / session_id

    def _session_upload_dir(self, session_id: str) -> Path:
        path = self.upload_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _read_text_from_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            if PdfReader is None:
                return ""
            reader = PdfReader(str(file_path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return file_path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _display_name(saved_path: Path) -> str:
        parts = saved_path.name.split("_", 1)
        return parts[1] if len(parts) == 2 else saved_path.name

    def _to_documents(self, file_path: Path, session_id: str) -> List[Document]:
        text = self._read_text_from_file(file_path).strip()
        if not text:
            return []
        return [Document(page_content=text, metadata={
            "source": str(file_path),
            "filename": self._display_name(file_path),
            "saved_filename": file_path.name,
            "session_id": session_id,
        })]

    def _load_store_from_disk(self, session_id: str) -> FAISS | None:
        index_path = self._session_index_path(session_id)
        if not index_path.exists():
            return None
        return FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _get_store(self, session_id: str) -> FAISS | None:
        cached = self._store_cache.get(session_id)
        if cached is not None:
            self._store_cache.move_to_end(session_id)
            return cached

        store = self._load_store_from_disk(session_id)
        if store is not None:
            self._store_cache[session_id] = store
            self._store_cache.move_to_end(session_id)
            while len(self._store_cache) > self._cache_max_size:
                self._store_cache.popitem(last=False)
        return store

    def _invalidate_store_cache(self, session_id: str) -> None:
        self._store_cache.pop(session_id, None)

    def _save_store(self, session_id: str, store: FAISS) -> None:
        index_path = self._session_index_path(session_id)
        index_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_path))
        self._store_cache[session_id] = store
        self._store_cache.move_to_end(session_id)
        while len(self._store_cache) > self._cache_max_size:
            self._store_cache.popitem(last=False)

    def _rebuild_index_from_saved_files(self, session_id: str) -> int:
        upload_dir = self._session_upload_dir(session_id)
        documents: List[Document] = []

        for saved_path in sorted(upload_dir.iterdir()):
            if not saved_path.is_file():
                continue
            docs = self._to_documents(saved_path, session_id=session_id)
            if docs:
                documents.extend(docs)

        index_path = self._session_index_path(session_id)
        if not documents:
            if index_path.exists():
                shutil.rmtree(index_path, ignore_errors=True)
            self._invalidate_store_cache(session_id)
            return 0

        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            if index_path.exists():
                shutil.rmtree(index_path, ignore_errors=True)
            self._invalidate_store_cache(session_id)
            return 0

        store = FAISS.from_documents(chunks, self.embeddings)
        self._save_store(session_id, store)
        return len(chunks)

    async def ingest_files(
        self,
        session_id: str,
        files: Iterable[UploadFile],
    ) -> Tuple[int, List[str]]:
        if not self.enabled:
            raise RuntimeError(self.disabled_reason)

        upload_dir = self._session_upload_dir(session_id)
        documents: List[Document] = []
        accepted_names: List[str] = []

        for upload in files:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in self.ALLOWED_SUFFIXES:
                continue

            safe_name = f"{uuid4().hex}_{Path(upload.filename).name}"
            saved_path = upload_dir / safe_name

            with saved_path.open("wb") as target:
                shutil.copyfileobj(upload.file, target)

            file_docs = self._to_documents(saved_path, session_id=session_id)
            if file_docs:
                documents.extend(file_docs)
                accepted_names.append(upload.filename)

        if not documents:
            return 0, accepted_names

        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            return 0, accepted_names

        store = self._get_store(session_id)
        if store is None:
            store = FAISS.from_documents(chunks, self.embeddings)
        else:
            store.add_documents(chunks)

        self._save_store(session_id, store)
        return len(chunks), accepted_names

    def retrieve_context(self, session_id: str, query: str, k: int | None = None) -> List[Document]:
        if not self.enabled:
            return []
        store = self._get_store(session_id)
        if store is None:
            return []
        return store.similarity_search(query, k=k or self.top_k)

    def citations_from_docs(self, docs: List[Document]) -> List[Dict[str, str]]:
        citations: List[Dict[str, str]] = []
        for idx, doc in enumerate(docs, 1):
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."
            citations.append({
                "id": str(idx),
                "filename": doc.metadata.get("filename", "unknown"),
                "snippet": snippet,
            })
        return citations

    def list_files(self, session_id: str) -> List[Dict[str, str]]:
        upload_dir = self._session_upload_dir(session_id)
        files: List[Dict[str, str]] = []
        for path in sorted(upload_dir.iterdir()):
            if not path.is_file():
                continue
            files.append({
                "saved_filename": path.name,
                "filename": self._display_name(path),
                "size_bytes": str(path.stat().st_size),
            })
        return files

    def delete_file(self, session_id: str, filename: str) -> bool:
        upload_dir = self._session_upload_dir(session_id)
        deleted = False
        for path in upload_dir.iterdir():
            if path.is_file() and self._display_name(path) == filename:
                path.unlink(missing_ok=True)
                deleted = True
        if deleted:
            self._rebuild_index_from_saved_files(session_id)
        return deleted

    def reindex_session(self, session_id: str) -> int:
        if not self.enabled:
            raise RuntimeError(self.disabled_reason)
        return self._rebuild_index_from_saved_files(session_id)


rag_service = RAGService()
