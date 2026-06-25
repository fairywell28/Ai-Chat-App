# coding: utf-8
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

from app.utils.time import utc_now
from config import get_config

Base = declarative_base()

_cfg = get_config()


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_session_created", "session_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    model_used = Column(String(50), default=_cfg.DEFAULT_LLM_MODEL)


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    temperature = Column(Integer, default=7)  # 0-10, 默认7
    max_tokens = Column(Integer, default=1000)
    model_preference = Column(String(50), default=_cfg.DEFAULT_LLM_MODEL)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )
