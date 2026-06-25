# coding: utf-8
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

from sqlalchemy.orm import Session

from app.models import Conversation, UserSettings
from app.services.openai_service import OpenAIService, openai_service
from app.services.rag_service import RAGService, rag_service
from app.services.session_service import get_llm_context_messages, parse_citations_json
from app.utils.time import utc_now, serialize_utc_datetime
from config import Config, get_config

logger = logging.getLogger(__name__)


def build_rag_messages(
    history_messages: List[Dict[str, str]],
    user_message: str,
    rag_context_docs: List[Any],
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


@dataclass
class PreparedChat:
    session_id: str
    user_message: str
    user_settings: UserSettings
    llm_messages: List[Dict[str, str]]
    citations: List[Dict[str, str]]


class ChatService:
    def __init__(
        self,
        cfg: Config,
        llm_service: OpenAIService,
        retrieval_service: RAGService,
    ) -> None:
        self.cfg = cfg
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service

    def resolve_session_id(self, session_id: str | None) -> str:
        return session_id or str(uuid.uuid4())

    def get_or_create_user_settings(
        self,
        db: Session,
        session_id: str,
    ) -> UserSettings:
        user_settings = (
            db.query(UserSettings)
            .filter(UserSettings.session_id == session_id)
            .first()
        )
        if user_settings:
            return user_settings

        user_settings = UserSettings(
            session_id=session_id,
            temperature=7,
            max_tokens=1000,
            model_preference=self.cfg.DEFAULT_LLM_MODEL,
        )
        db.add(user_settings)
        db.commit()
        db.refresh(user_settings)
        return user_settings

    def prepare_chat(
        self,
        db: Session,
        session_id: str,
        user_message: str,
    ) -> PreparedChat:
        user_settings = self.get_or_create_user_settings(db, session_id)
        history = get_llm_context_messages(
            db,
            session_id,
            turn_limit=self.cfg.LLM_CONTEXT_TURN_LIMIT,
        )
        rag_docs = self.retrieval_service.retrieve_context(
            session_id=session_id,
            query=user_message,
            k=self.cfg.RAG_TOP_K,
        )
        citations = self.retrieval_service.citations_from_docs(rag_docs)
        llm_messages = build_rag_messages(history, user_message, rag_docs)
        return PreparedChat(
            session_id=session_id,
            user_message=user_message,
            user_settings=user_settings,
            llm_messages=llm_messages,
            citations=citations,
        )

    def save_conversation_turn(
        self,
        db: Session,
        prepared: PreparedChat,
        ai_response: str,
    ) -> Conversation:
        conversation = Conversation(
            session_id=prepared.session_id,
            user_message=prepared.user_message,
            ai_response=ai_response,
            citations_json=json.dumps(prepared.citations) if prepared.citations else None,
            model_used=prepared.user_settings.model_preference,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    async def complete_message(
        self,
        db: Session,
        session_id: str | None,
        user_message: str,
    ) -> Dict[str, Any]:
        resolved_session_id = self.resolve_session_id(session_id)
        prepared = self.prepare_chat(db, resolved_session_id, user_message)

        logger.debug(
            "messages=%s, model=%s, temperature=%s, max_tokens=%s",
            prepared.llm_messages,
            prepared.user_settings.model_preference,
            prepared.user_settings.temperature / 10.0,
            prepared.user_settings.max_tokens,
        )

        response = await self.llm_service.chat_completion(
            messages=prepared.llm_messages,
            model=prepared.user_settings.model_preference,
            temperature=prepared.user_settings.temperature / 10.0,
            max_tokens=prepared.user_settings.max_tokens,
        )
        self.save_conversation_turn(db, prepared, response)

        return {
            "session_id": prepared.session_id,
            "user_message": prepared.user_message,
            "ai_response": response,
            "citations": prepared.citations,
            "timestamp": serialize_utc_datetime(utc_now()),
        }

    async def stream_message(
        self,
        db: Session,
        session_id: str | None,
        user_message: str,
    ) -> AsyncIterator[str]:
        resolved_session_id = self.resolve_session_id(session_id)
        prepared = self.prepare_chat(db, resolved_session_id, user_message)

        yield json.dumps({"citations": prepared.citations})

        full_response = ""
        async for chunk in self.llm_service.stream_chat_completion(
            messages=prepared.llm_messages,
            model=prepared.user_settings.model_preference,
            temperature=prepared.user_settings.temperature / 10.0,
            max_tokens=prepared.user_settings.max_tokens,
        ):
            full_response += chunk
            yield json.dumps({"content": chunk})

        self.save_conversation_turn(db, prepared, full_response)


chat_service = ChatService(get_config(), openai_service, rag_service)
