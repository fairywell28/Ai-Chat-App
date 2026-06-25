# coding: utf-8
"""
Phase 1 vs previous version — local performance benchmarks.

Run: ./venv/bin/python benchmarks/phase1_comparison.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Conversation
from app.utils.time import utc_now

Base = declarative_base()


class ConversationNoIndex(Base):
    """Previous schema: session_id index only, no composite index."""
    __tablename__ = "conversations_legacy"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    model_used = Column(String(50), default="gpt-4.1-mini")


class ConversationWithIndex(Base):
    """Current schema: composite (session_id, created_at) index."""
    __tablename__ = "conversations_new"
    __table_args__ = (Index("ix_conversations_session_created", "session_id", "created_at"),)

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    model_used = Column(String(50), default="gpt-4.1-mini")


def _timeit(fn, rounds: int = 200) -> dict:
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(len(samples) * 0.95) - 1],
        "min_ms": min(samples),
    }


def _seed_conversations(model, db, sessions: int, turns_per_session: int) -> None:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    citations = json.dumps([{"id": "1", "filename": "doc.txt", "snippet": "x" * 120}])
    for s in range(sessions):
        sid = f"session-{s:03d}"
        for t in range(turns_per_session):
            db.add(
                model(
                    session_id=sid,
                    user_message=f"question {s}-{t}",
                    ai_response=f"answer {s}-{t}",
                    citations_json=citations,
                    created_at=base + timedelta(minutes=s * turns_per_session + t),
                )
            )
    db.commit()


def benchmark_db_queries() -> dict:
    engine_legacy = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    engine_new = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine_legacy, tables=[ConversationNoIndex.__table__])
    Base.metadata.create_all(bind=engine_new, tables=[ConversationWithIndex.__table__])

    legacy_session = sessionmaker(bind=engine_legacy)()
    new_session = sessionmaker(bind=engine_new)()

    _seed_conversations(ConversationNoIndex, legacy_session, sessions=80, turns_per_session=25)
    _seed_conversations(ConversationWithIndex, new_session, sessions=80, turns_per_session=25)

    target = "session-040"

    def legacy_context():
        rows = (
            legacy_session.query(ConversationNoIndex)
            .filter(ConversationNoIndex.session_id == target)
            .order_by(ConversationNoIndex.created_at.desc())
            .limit(10)
            .all()
        )
        return list(reversed(rows))

    def new_context():
        rows = (
            new_session.query(ConversationWithIndex)
            .filter(ConversationWithIndex.session_id == target)
            .order_by(ConversationWithIndex.created_at.desc())
            .limit(10)
            .all()
        )
        return list(reversed(rows))

    def legacy_history():
        rows = (
            legacy_session.query(ConversationNoIndex)
            .filter(ConversationNoIndex.session_id == target)
            .order_by(ConversationNoIndex.created_at.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "user_message": r.user_message,
                "ai_response": r.ai_response,
                "citations": json.loads(r.citations_json or "[]"),
            }
            for r in reversed(rows)
        ]

    def new_history():
        rows = (
            new_session.query(ConversationWithIndex)
            .filter(ConversationWithIndex.session_id == target)
            .order_by(ConversationWithIndex.created_at.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "user_message": r.user_message,
                "ai_response": r.ai_response,
                "citations": json.loads(r.citations_json or "[]"),
            }
            for r in reversed(rows)
        ]

    def legacy_sessions():
        from sqlalchemy import func

        aggregates = (
            legacy_session.query(
                ConversationNoIndex.session_id,
                func.max(ConversationNoIndex.created_at),
                func.count(ConversationNoIndex.id),
            )
            .group_by(ConversationNoIndex.session_id)
            .order_by(func.max(ConversationNoIndex.created_at).desc())
            .limit(10)
            .all()
        )
        for row in aggregates:
            legacy_session.query(ConversationNoIndex).filter(
                ConversationNoIndex.session_id == row[0]
            ).order_by(ConversationNoIndex.created_at.desc()).first()

    def new_sessions():
        from sqlalchemy import func

        aggregates = (
            new_session.query(
                ConversationWithIndex.session_id,
                func.max(ConversationWithIndex.created_at),
                func.count(ConversationWithIndex.id),
            )
            .group_by(ConversationWithIndex.session_id)
            .order_by(func.max(ConversationWithIndex.created_at).desc())
            .limit(10)
            .all()
        )
        for row in aggregates:
            new_session.query(ConversationWithIndex).filter(
                ConversationWithIndex.session_id == row[0]
            ).order_by(ConversationWithIndex.created_at.desc()).first()

    legacy_plan = legacy_session.execute(
        text(
            "EXPLAIN QUERY PLAN SELECT * FROM conversations_legacy "
            "WHERE session_id = :sid ORDER BY created_at DESC LIMIT 10"
        ),
        {"sid": target},
    ).fetchall()
    new_plan = new_session.execute(
        text(
            "EXPLAIN QUERY PLAN SELECT * FROM conversations_new "
            "WHERE session_id = :sid ORDER BY created_at DESC LIMIT 10"
        ),
        {"sid": target},
    ).fetchall()

    return {
        "llm_context_legacy": _timeit(legacy_context),
        "llm_context_new": _timeit(new_context),
        "history_legacy": _timeit(legacy_history),
        "history_new": _timeit(new_history),
        "sessions_legacy": _timeit(legacy_sessions, rounds=50),
        "sessions_new": _timeit(new_sessions, rounds=50),
        "explain_legacy": [row[3] for row in legacy_plan],
        "explain_new": [row[3] for row in new_plan],
    }


def benchmark_rag_cache() -> dict:
    """Simulate old (disk every call) vs new (memory cache hit)."""
    load_ms = 12.0  # simulated FAISS.load_local cost per call

    def old_retrieve(rounds: int = 100):
        total = 0.0
        for _ in range(rounds):
            start = time.perf_counter()
            time.sleep(load_ms / 1000.0)
            total += time.perf_counter() - start
        return (total / rounds) * 1000

    cache: OrderedDict[str, MagicMock] = OrderedDict()
    fake_store = MagicMock()
    fake_store.similarity_search.return_value = []

    def new_retrieve_first():
        start = time.perf_counter()
        time.sleep(load_ms / 1000.0)
        cache["s1"] = fake_store
        return (time.perf_counter() - start) * 1000

    def new_retrieve_cached():
        start = time.perf_counter()
        store = cache.get("s1")
        store.similarity_search("query", k=4)
        return (time.perf_counter() - start) * 1000

    old_mean = old_retrieve()
    first_hit = new_retrieve_first()
    cached_samples = [_timeit(new_retrieve_cached, rounds=1)["mean_ms"] for _ in range(100)]
    cached_mean = statistics.mean(cached_samples)

    return {
        "old_every_call_ms": old_mean,
        "new_first_load_ms": first_hit,
        "new_cached_hit_ms": cached_mean,
        "speedup_cached_vs_old": round(old_mean / max(cached_mean, 0.001), 1),
    }


def benchmark_write_overhead() -> dict:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[Conversation.__table__])
    db = sessionmaker(bind=engine)()

    def write_without_citations():
        db.add(
            Conversation(
                session_id="w1",
                user_message="hi",
                ai_response="hello",
                citations_json=None,
            )
        )
        db.commit()
        db.query(Conversation).delete()
        db.commit()

    def write_with_citations():
        db.add(
            Conversation(
                session_id="w1",
                user_message="hi",
                ai_response="hello",
                citations_json=json.dumps(
                    [{"id": "1", "filename": "a.txt", "snippet": "snippet"}]
                ),
            )
        )
        db.commit()
        db.query(Conversation).delete()
        db.commit()

    return {
        "write_no_citations": _timeit(write_without_citations, rounds=100),
        "write_with_citations": _timeit(write_with_citations, rounds=100),
    }


def _pct_change(old: float, new: float) -> str:
    if old == 0:
        return "n/a"
    delta = ((new - old) / old) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def main() -> None:
    print("=" * 72)
    print("AI Chat App — Phase 1 vs Previous Version Performance Comparison")
    print("Environment: local benchmark, SQLite in-memory, mocked RAG disk load")
    print("=" * 72)

    db = benchmark_db_queries()
    rag = benchmark_rag_cache()
    write = benchmark_write_overhead()

    print("\n## 1. Database query latency (80 sessions × 25 turns)")
    print(f"{'Operation':<28} {'Previous':>12} {'Phase 1':>12} {'Change':>10}")
    print("-" * 72)
    for label, old_key, new_key in [
        ("LLM context (10 turns)", "llm_context_legacy", "llm_context_new"),
        ("History API (50 turns)", "history_legacy", "history_new"),
        ("Session list (10 items)", "sessions_legacy", "sessions_new"),
    ]:
        old = db[old_key]["mean_ms"]
        new = db[new_key]["mean_ms"]
        print(f"{label:<28} {old:>10.3f} ms {new:>10.3f} ms {_pct_change(old, new):>10}")

    print("\nSQLite EXPLAIN (context query):")
    print(f"  Previous: {db['explain_legacy']}")
    print(f"  Phase 1:  {db['explain_new']}")

    print("\n## 2. RAG retrieval (simulated FAISS load = 12 ms)")
    print(f"  Previous (load disk every call): {rag['old_every_call_ms']:.2f} ms")
    print(f"  Phase 1 first load:              {rag['new_first_load_ms']:.2f} ms")
    print(f"  Phase 1 cache hit:               {rag['new_cached_hit_ms']:.4f} ms")
    print(f"  Cache hit speedup vs previous:   ~{rag['speedup_cached_vs_old']}×")

    print("\n## 3. Write overhead (citation persistence)")
    old_w = write["write_no_citations"]["mean_ms"]
    new_w = write["write_with_citations"]["mean_ms"]
    print(f"  Previous (no citations_json): {old_w:.3f} ms")
    print(f"  Phase 1 (with citations_json): {new_w:.3f} ms ({_pct_change(old_w, new_w)})")

    print("\n## 4. Items with negligible runtime impact")
    print("  - ChatService refactor (code dedup, same call chain)")
    print("  - Config unification (read at startup)")
    print("  - timezone-aware utc_now()")

    print("\n## 5. Summary")
    ctx_old, ctx_new = db["llm_context_legacy"]["mean_ms"], db["llm_context_new"]["mean_ms"]
    if ctx_new <= ctx_old:
        print(f"  DB context query: improved or similar ({_pct_change(ctx_old, ctx_new)})")
    else:
        print(f"  DB context query: slightly slower ({_pct_change(ctx_old, ctx_new)}), acceptable for correctness fixes")

    print(
        f"  RAG hot path: major win on repeated queries in same session "
        f"(~{rag['speedup_cached_vs_old']}× faster after first load)"
    )
    print(f"  Citation persist: tiny write cost ({_pct_change(old_w, new_w)})")
    print("  Session list N+1: unchanged (Phase 2 optimization)")
    print("=" * 72)


if __name__ == "__main__":
    main()
