# coding: utf-8
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.models import Base
from config import get_config

_cfg = get_config()

engine = create_engine(
    _cfg.DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _ensure_conversations_citations_column() -> None:
    """Lightweight schema patch for existing SQLite databases."""
    inspector = inspect(engine)
    if "conversations" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("conversations")}
    if "citations_json" in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE conversations ADD COLUMN citations_json TEXT")
        )


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_conversations_citations_column()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
