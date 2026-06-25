# coding: utf-8
import json
import logging

from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.chat_service import chat_service
from app.services.rag_service import rag_service
from app.services.session_service import (
    DEFAULT_SESSION_LIMIT,
    get_session_history,
    list_recent_sessions,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@router.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    if not rag_service.enabled:
        raise HTTPException(status_code=503, detail=rag_service.disabled_reason)

    try:
        chunk_count, accepted_files = await rag_service.ingest_files(session_id, files)
        if not accepted_files:
            raise HTTPException(
                status_code=400,
                detail="未检测到可支持的文件类型，请上传 txt/md/csv/json/pdf/py/log 文件。",
            )
        return {
            "session_id": session_id,
            "uploaded_files": accepted_files,
            "indexed_chunks": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG文件上传处理失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{session_id}")
async def list_uploaded_files(session_id: str):
    if not rag_service.enabled:
        raise HTTPException(status_code=503, detail=rag_service.disabled_reason)
    return {
        "session_id": session_id,
        "files": rag_service.list_files(session_id),
    }


@router.delete("/files/{session_id}")
async def delete_uploaded_file(session_id: str, filename: str):
    if not rag_service.enabled:
        raise HTTPException(status_code=503, detail=rag_service.disabled_reason)
    deleted = rag_service.delete_file(session_id, filename)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    return {"session_id": session_id, "deleted_file": filename}


@router.post("/files/{session_id}/reindex")
async def reindex_files(session_id: str):
    if not rag_service.enabled:
        raise HTTPException(status_code=503, detail=rag_service.disabled_reason)
    try:
        chunks = rag_service.reindex_session(session_id)
        return {"session_id": session_id, "indexed_chunks": chunks}
    except Exception as e:
        logger.exception("RAG重建索引失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
async def send_message(payload: ChatRequest, db: Session = Depends(get_db)):
    try:
        return await chat_service.complete_message(
            db=db,
            session_id=payload.session_id,
            user_message=payload.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/stream")
async def stream_message(payload: ChatRequest, db: Session = Depends(get_db)):
    async def generate():
        try:
            async for event in chat_service.stream_message(
                db=db,
                session_id=payload.session_id,
                user_message=payload.message,
            ):
                yield f"data: {event}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/sessions")
async def get_recent_sessions(
    limit: int = DEFAULT_SESSION_LIMIT,
    db: Session = Depends(get_db),
):
    safe_limit = max(1, min(limit, 50))
    return {
        "sessions": list_recent_sessions(db, limit=safe_limit),
    }


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    return get_session_history(db, session_id, limit=limit)
