"""SQLite connection manager and schema initialization."""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DB_DIR = "data/memory"
DB_PATH = os.path.join(DB_DIR, "agent_memory.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    skill           TEXT,
    query_type      TEXT,
    result_summary  TEXT,
    intent_tags     TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conv_session
    ON conversation_turns(session_id, turn_index);

CREATE TABLE IF NOT EXISTS retrieval_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    turn_index    INTEGER NOT NULL,
    query         TEXT    NOT NULL,
    skill         TEXT    NOT NULL,
    top_k         INTEGER DEFAULT 20,
    filters       TEXT,
    result_count  INTEGER DEFAULT 0,
    top_results   TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_retrieval_session
    ON retrieval_snapshots(session_id);

CREATE TABLE IF NOT EXISTS user_preferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL UNIQUE,
    preferred_skill  TEXT    DEFAULT 'tdoc',
    working_groups   TEXT    DEFAULT '[]',
    companies        TEXT    DEFAULT '[]',
    top_k            INTEGER DEFAULT 20,
    language         TEXT    DEFAULT 'zh-CN',
    updated_at       TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS search_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    query           TEXT    NOT NULL,
    skill           TEXT    NOT NULL,
    filters         TEXT,
    result_count    INTEGER DEFAULT 0,
    clicked_doc_ids  TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_search_user
    ON search_history(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_search_query
    ON search_history(query);

CREATE TABLE IF NOT EXISTS bookmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    doc_id      TEXT    NOT NULL,
    skill       TEXT    NOT NULL,
    title       TEXT,
    metadata    TEXT,
    note        TEXT,
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(user_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_bookmarks_user
    ON bookmarks(user_id);

CREATE TABLE IF NOT EXISTS intent_tracking (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    topic       TEXT    NOT NULL,
    confidence  REAL    DEFAULT 0.0,
    queries     TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now')),
    expires_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_intent_user
    ON intent_tracking(user_id, created_at DESC);
"""


class MemoryDB:
    """Thread-safe SQLite connection manager for agent memory."""

    _instance: Optional["MemoryDB"] = None

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @classmethod
    def get_instance(cls, db_path: str = DB_PATH) -> "MemoryDB":
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def _init_schema(self):
        conn = self.get_connection()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            logger.info(f"Memory DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to init memory schema: {e}")
            raise
        finally:
            conn.close()

    def get_connection(self):
        conn = __import__("sqlite3").connect(self.db_path)
        conn.row_factory = __import__("sqlite3").Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def purge_expired(self, max_age_hours: int = 48):
        """Delete old conversation/retrieval rows from inactive sessions."""
        cutoff = (
            datetime.now() - timedelta(hours=max_age_hours)
        ).isoformat()
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM retrieval_snapshots WHERE created_at < ?",
                (cutoff,),
            )
            cur.execute(
                "DELETE FROM conversation_turns WHERE created_at < ?",
                (cutoff,),
            )
            cur.execute(
                "DELETE FROM intent_tracking WHERE expires_at IS NOT NULL AND expires_at < ?",
                (cutoff,),
            )
            conn.commit()
            logger.info("Expired memory rows purged")
        finally:
            conn.close()
