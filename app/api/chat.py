# coding: utf-8
from typing import List, Dict
from datetime import datetime
import uuid
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, UserSettings
from app.services.openai_service import openai_service
from app.services.rag_service import rag_service
from config import get_config

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])
cfg = get_config()

# setting the default LLM model, @TODO setting by config file in future
default_llm = "gpt-4.1-mini"

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# @SampleCode Building RAG messages for LLM: history + current + RAG
def build_rag_messages(
    history_messages: List[Dict[str, str]],
    user_message: str,
    rag_context_docs: List[Dict[str, str]] | List
) -> List[Dict[str, str]]:
    if not rag_context_docs:
        return history_messages + [{"role": "user", "content": user_message}]

    context_blocks = []
    for idx, doc in enumerate(rag_context_docs, 1):
        source = doc.metadata.get("filename", "unknown")
        context_blocks.append(
            f"[资料{idx} | 来源: {source}]\n{doc.page_content}"
        )

    rag_instruction = (
        "你是专业的AI助手。回答用户问题时，请优先依据提供的资料内容。"
        "如果资料中无法支持结论，请明确说明并给出通用建议，不要编造来源。"
        "\n\n"
        "以下是可参考资料：\n"
        f"{chr(10).join(context_blocks)}"
    )
    return [{"role": "system", "content": rag_instruction}] + history_messages + [
        {"role": "user", "content": user_message}
    ]


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
    """
    发送消息并获取AI回复
    """
    # 从请求体中获取数据并生成/使用session_id
    message = payload.message
    session_id = payload.session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    # 获取用户设置
    user_settings = db.query(UserSettings).filter(
        UserSettings.session_id == session_id
    ).first()

    if not user_settings:
        # 创建默认用户设置
        user_settings = UserSettings(
            session_id=session_id,
            temperature=7,
            max_tokens=1000,
            model_preference=default_llm
        )
        db.add(user_settings)
        db.commit()
        db.refresh(user_settings)

    # 构建历史对话
    conversation_history = await get_conversation_history(session_id, db)

    # 添加用户新消息（结合RAG检索）
    rag_docs = rag_service.retrieve_context(
        session_id=session_id,
        query=message,
        k=cfg.RAG_TOP_K,
    )
    citations = rag_service.citations_from_docs(rag_docs)
    messages = build_rag_messages(conversation_history, message, rag_docs)

    try:
        # 调用OpenAI服务
        logger.debug(f"messages={messages}, model={user_settings.model_preference}, temperature={user_settings.temperature/10.0}, max_tokens={user_settings.max_tokens}")
        logger.debug("Start getting response...")
        response = await openai_service.chat_completion(
            messages=messages,
            model=user_settings.model_preference,
            temperature=user_settings.temperature / 10.0,  # 转换为0-1范围
            max_tokens=user_settings.max_tokens
        )
        logger.debug("Finished getting the response.")

        # 保存对话记录
        conversation = Conversation(
            session_id=session_id,
            user_message=message,
            ai_response=response,
            model_used=user_settings.model_preference
        )
        db.add(conversation)
        db.commit()

        return {
            "session_id": session_id,
            "user_message": message,
            "ai_response": response,
            "citations": citations,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/stream")
async def stream_message(
    payload: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    流式消息响应（实时打字效果）
    """
    message = payload.message
    session_id = payload.session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    # 获取用户设置
    user_settings = db.query(UserSettings).filter(
        UserSettings.session_id == session_id
    ).first()

    if not user_settings:
        # 创建默认用户设置
        user_settings = UserSettings(
            session_id=session_id,
            temperature=7,
            max_tokens=1000,
            model_preference=default_llm
        )
        db.add(user_settings)
        db.commit()
        db.refresh(user_settings)

    # 构建历史对话
    conversation_history = await get_conversation_history(session_id, db)

    # 添加用户新消息（结合RAG检索）
    rag_docs = rag_service.retrieve_context(
        session_id=session_id,
        query=message,
        k=cfg.RAG_TOP_K,
    )
    citations = rag_service.citations_from_docs(rag_docs)
    messages = build_rag_messages(conversation_history, message, rag_docs)

    async def generate():
        try:
            full_response = ""
            yield f"data: {json.dumps({'citations': citations})}\n\n"
            async for chunk in openai_service.stream_chat_completion(
                messages=messages,
                model=user_settings.model_preference,
                temperature=user_settings.temperature / 10.0,
                max_tokens=user_settings.max_tokens
            ):
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # 保存完整对话记录
            conversation = Conversation(
                session_id=session_id,
                user_message=message,
                ai_response=full_response,
                model_used=user_settings.model_preference
            )
            db.add(conversation)
            db.commit()

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    获取指定会话的聊天历史
    """
    conversations = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).order_by(Conversation.created_at.desc()).limit(limit).all()

    return [
        {
            "id": conv.id,
            "user_message": conv.user_message,
            "ai_response": conv.ai_response,
            "created_at": conv.created_at.isoformat(),
            "model_used": conv.model_used
        }
        for conv in reversed(conversations)  # 按时间正序返回
    ]


async def get_conversation_history(session_id: str,
                                   db: Session) -> List[Dict[str, str]]:
    """
    获取格式化后的对话历史
    """
    conversations = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).order_by(Conversation.created_at.asc()).limit(10).all()  # 最近10条对话

    messages = []
    for conv in conversations:
        messages.append({"role": "user", "content": conv.user_message})
        messages.append({"role": "assistant", "content": conv.ai_response})

    return messages
