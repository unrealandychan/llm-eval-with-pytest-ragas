"""
memory.py — Two-tier memory layer for LLM evaluation pipelines.

Short-term memory  — in-process circular buffer; holds the most recent
                     conversation turns for the current session.

Long-term memory   — SQLite-backed persistent store; survives restarts and
                     keeps a full history of turns, evaluation results, and
                     facts across sessions.

Unified interface  — MemoryStore wires both tiers together and exposes a
                     simple API that the RAG pipeline can consume.

Usage (no API key required)::

    from llm_eval.memory import MemoryStore

    mem = MemoryStore()                  # ephemeral in-memory DB
    mem = MemoryStore("my_project.db")   # persistent SQLite file

    mem.add_turn("user", "What is the GIL?")
    mem.add_turn("assistant", "The GIL is a mutex …")

    for entry in mem.get_recent_turns(5):
        print(entry.role, entry.content[:80])

    mem.add_eval_result(
        question="What is the GIL?",
        answer="The GIL is a mutex …",
        scores={"faithfulness": 0.9, "agentic": 0.85},
    )

    for r in mem.search("GIL", entry_type="eval_result"):
        print(r.content)

    mem.close()
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single stored memory item."""

    session_id: str
    entry_type: str         # "turn" | "eval_result" | "fact"
    content: str
    role: str | None = None  # "user" | "assistant" — relevant for turns
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    id: int | None = None    # assigned by the DB after INSERT

    def metadata_json(self) -> str:
        return json.dumps(self.metadata)


# ---------------------------------------------------------------------------
# Short-term memory  (in-process, circular buffer)
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """
    Fixed-capacity in-memory buffer for the current session's recent turns.

    Only 'turn' entries (user / assistant messages) are kept here.
    Older entries are evicted automatically once the buffer is full.
    """

    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._buffer: deque[MemoryEntry] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, entry: MemoryEntry) -> None:
        self._buffer.append(entry)

    def get_recent(self, n: int | None = None) -> list[MemoryEntry]:
        """Return the n most recent entries (default: all in buffer)."""
        items = list(self._buffer)
        if n is not None:
            items = items[-n:]
        return items

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Long-term memory  (SQLite-backed)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    entry_type  TEXT    NOT NULL,
    role        TEXT,
    content     TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  REAL    NOT NULL
);
"""

_CREATE_IDX_SESSION = (
    "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);"
)
_CREATE_IDX_TYPE = (
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(entry_type);"
)


class LongTermMemory:
    """
    SQLite-backed persistent store.

    Pass ``db_path=":memory:"`` (the default) for a throwaway in-process
    database — useful in tests.  Pass a file path for durable storage.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_TABLE)
        cur.execute(_CREATE_IDX_SESSION)
        cur.execute(_CREATE_IDX_TYPE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, entry: MemoryEntry) -> MemoryEntry:
        """Insert *entry* into the DB and return it with its new ``id``."""
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memories (session_id, entry_type, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.session_id,
                entry.entry_type,
                entry.role,
                entry.content,
                entry.metadata_json(),
                entry.created_at,
            ),
        )
        self._conn.commit()
        entry.id = cur.lastrowid
        return entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str, limit: int | None = None) -> list[MemoryEntry]:
        """All conversation turns for *session_id*, oldest first."""
        sql = (
            "SELECT * FROM memories "
            "WHERE session_id=? AND entry_type='turn' "
            "ORDER BY created_at ASC"
        )
        params: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._fetch(sql, params)

    def get_all_by_type(
        self,
        entry_type: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return up to *limit* entries of a given type, newest first."""
        return self._fetch(
            "SELECT * FROM memories WHERE entry_type=? ORDER BY created_at DESC LIMIT ?",
            [entry_type, limit],
        )

    def search(
        self,
        query: str,
        entry_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """
        Simple full-text substring search over the ``content`` column.

        For production use, consider replacing this with FTS5 or an embedding
        similarity search on top of the stored metadata.
        """
        pattern = f"%{query}%"
        if entry_type:
            return self._fetch(
                "SELECT * FROM memories "
                "WHERE entry_type=? AND content LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                [entry_type, pattern, limit],
            )
        return self._fetch(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
            [pattern, limit],
        )

    def count(self, entry_type: str | None = None) -> int:
        cur = self._conn.cursor()
        if entry_type:
            cur.execute("SELECT COUNT(*) FROM memories WHERE entry_type=?", [entry_type])
        else:
            cur.execute("SELECT COUNT(*) FROM memories")
        row = cur.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch(self, sql: str, params: list[Any]) -> list[MemoryEntry]:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return [self._row_to_entry(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            session_id=row["session_id"],
            entry_type=row["entry_type"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Unified MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """
    Unified two-tier memory store.

    Example — persistent file::

        store = MemoryStore("eval_memory.db")
        store.add_turn("user", "What is the GIL?")

    Example — ephemeral (tests / CI)::

        store = MemoryStore()   # defaults to in-process SQLite

    Closing::

        store.close()

    Or use as a context manager::

        with MemoryStore("eval_memory.db") as store:
            store.add_turn("user", "...")
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        session_id: str | None = None,
        short_term_capacity: int = 20,
    ):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(db_path=db_path)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Record one conversation turn.

        Saved to *both* short-term (evictable) and long-term (persistent)
        memory.
        """
        entry = MemoryEntry(
            session_id=self.session_id,
            entry_type="turn",
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.long_term.save(entry)
        self.short_term.add(entry)
        return entry

    def add_eval_result(
        self,
        question: str,
        answer: str,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Persist an evaluation result to long-term memory only.

        ``content`` is a compact representation; full details live in
        ``metadata``.
        """
        extra = dict(metadata or {})
        extra.update({"question": question, "answer": answer, "scores": scores})
        content = f"Q: {question[:120]} | scores: {scores}"
        entry = MemoryEntry(
            session_id=self.session_id,
            entry_type="eval_result",
            content=content,
            metadata=extra,
        )
        return self.long_term.save(entry)

    def add_fact(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Persist a free-form fact to long-term memory."""
        entry = MemoryEntry(
            session_id=self.session_id,
            entry_type="fact",
            content=content,
            metadata=metadata or {},
        )
        return self.long_term.save(entry)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_recent_turns(self, n: int | None = None) -> list[MemoryEntry]:
        """Most recent turns from the *short-term* buffer."""
        return self.short_term.get_recent(n)

    def get_session_history(self, session_id: str | None = None) -> list[MemoryEntry]:
        """All persisted turns for *session_id* (defaults to current session)."""
        return self.long_term.get_session_turns(session_id or self.session_id)

    def search(
        self,
        query: str,
        entry_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Substring search across long-term memory."""
        return self.long_term.search(query, entry_type=entry_type, limit=limit)

    def get_eval_results(self, limit: int = 50) -> list[MemoryEntry]:
        """Return recent evaluation results from long-term memory."""
        return self.long_term.get_all_by_type("eval_result", limit=limit)

    def get_facts(self, limit: int = 50) -> list[MemoryEntry]:
        """Return stored facts from long-term memory."""
        return self.long_term.get_all_by_type("fact", limit=limit)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.long_term.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
