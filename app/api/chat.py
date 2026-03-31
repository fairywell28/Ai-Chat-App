# coding: utf-8
from typing import List, Dict
from datetime import datetime
import uuid
import json
import logging

from fastapi import APIRouter, Depends, Response, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, UserSettings
from app.services.openai_service import openai_service

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# setting the default LLM model, @TODO setting by config file in future
default_llm = "gpt-4.1-mini"

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


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

    # 添加用户新消息
    messages = conversation_history + [{"role": "user", "content": message}]

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

    # 添加用户新消息
    messages = conversation_history + [{"role": "user", "content": message}]

    async def generate():
        try:
            full_response = ""
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
