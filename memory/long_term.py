"""Long-term user memory (SQLite-backed)."""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from .db import MemoryDB
from .models import Bookmark, IntentTopic, SearchHistoryEntry, UserPreferences

logger = logging.getLogger(__name__)

_STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
              "of", "with", "by", "is", "are", "was", "were", "be", "been",
              "this", "that", "这些", "这个", "关于", "的", "了", "在", "和"}


class UserMemory:
    """Cross-session user memory backed by SQLite.

    One instance per user (or 'default' for anonymous).
    Thread-safety: each call gets its own DB connection (SQLite WAL mode).
    """

    def __init__(self, user_id: str, db: MemoryDB):
        self.user_id = user_id
        self._db = db

    # ── Preferences ──────────────────────────────────────────────────────────────

    def load_preferences(self) -> UserPreferences:
        """Load user preferences from SQLite. Returns defaults if none exist."""
        conn = self._db.get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()
            if not row:
                return UserPreferences(user_id=self.user_id)

            return UserPreferences(
                user_id=row["user_id"],
                preferred_skill=row["preferred_skill"] or "tdoc",
                working_groups=json.loads(row["working_groups"] or "[]"),
                companies=json.loads(row["companies"] or "[]"),
                top_k=row["top_k"] or 20,
                language=row["language"] or "zh-CN",
            )
        finally:
            conn.close()

    def save_preferences(self, prefs: UserPreferences):
        """Upsert user preferences."""
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO user_preferences
                      (user_id, preferred_skill, working_groups, companies, top_k, language, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                      preferred_skill=excluded.preferred_skill,
                      working_groups=excluded.working_groups,
                      companies=excluded.companies,
                      top_k=excluded.top_k,
                      language=excluded.language,
                      updated_at=excluded.updated_at""",
                (
                    prefs.user_id,
                    prefs.preferred_skill,
                    json.dumps(prefs.working_groups, ensure_ascii=False),
                    json.dumps(prefs.companies, ensure_ascii=False),
                    prefs.top_k,
                    prefs.language,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_preference(self, key: str, value):
        prefs = self.load_preferences()
        if hasattr(prefs, key):
            setattr(prefs, key, value)
        self.save_preferences(prefs)

    # ── Search history ──────────────────────────────────────────────────────────

    def record_search(self, entry: SearchHistoryEntry):
        """Record a search query in history."""
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO search_history
                   (user_id, session_id, query, skill, filters, result_count,
                    clicked_doc_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.user_id,
                    entry.session_id,
                    entry.query,
                    entry.skill,
                    json.dumps(entry.filters, ensure_ascii=False)
                    if entry.filters
                    else None,
                    entry.result_count,
                    json.dumps(entry.clicked_doc_ids, ensure_ascii=False)
                    if entry.clicked_doc_ids
                    else None,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_searches(self, limit: int = 20) -> list[SearchHistoryEntry]:
        """Get recent searches for this user."""
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT * FROM search_history
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (self.user_id, limit),
            ).fetchall()
            return [
                SearchHistoryEntry(
                    user_id=r["user_id"],
                    session_id=r["session_id"],
                    query=r["query"],
                    skill=r["skill"],
                    filters=json.loads(r["filters"]) if r["filters"] else None,
                    result_count=r["result_count"],
                    clicked_doc_ids=(
                        json.loads(r["clicked_doc_ids"])
                        if r["clicked_doc_ids"]
                        else None
                    ),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_topic_frequency(self, days: int = 30) -> dict[str, int]:
        """Return {topic: count} for the most searched topics in the last N days."""
        cutoff = (
            datetime.now() - timedelta(days=days)
        ).isoformat()
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT query FROM search_history
                   WHERE user_id = ? AND created_at > ?
                   ORDER BY created_at DESC""",
                (self.user_id, cutoff),
            ).fetchall()
            freq: dict[str, int] = {}
            for r in rows:
                words = re.findall(r"[a-zA-Z0-9一-鿿]{3,}", r["query"].lower())
                for w in words:
                    if w not in _STOPWORDS:
                        freq[w] = freq.get(w, 0) + 1
            return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20])
        finally:
            conn.close()

    def get_frequent_companies(self, limit: int = 5) -> list[str]:
        """Return companies most frequently filtered for."""
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT filters FROM search_history
                   WHERE user_id = ? AND filters LIKE '%company%'
                   ORDER BY created_at DESC
                   LIMIT 100""",
                (self.user_id,),
            ).fetchall()
            company_counts: dict[str, int] = {}
            for r in rows:
                f = json.loads(r["filters"]) if r["filters"] else {}
                if f.get("company"):
                    company_counts[f["company"]] = (
                        company_counts.get(f["company"], 0) + 1
                    )
            return [c for c, _ in sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:limit]]
        finally:
            conn.close()

    def get_frequent_working_groups(self, limit: int = 5) -> list[str]:
        """Return WGs most frequently filtered for."""
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT filters FROM search_history
                   WHERE user_id = ? AND filters LIKE '%working_group%'
                   ORDER BY created_at DESC
                   LIMIT 100""",
                (self.user_id,),
            ).fetchall()
            wg_counts: dict[str, int] = {}
            for r in rows:
                f = json.loads(r["filters"]) if r["filters"] else {}
                if f.get("working_group"):
                    wg_counts[f["working_group"]] = (
                        wg_counts.get(f["working_group"], 0) + 1
                    )
            return [wg for wg, _ in sorted(wg_counts.items(), key=lambda x: x[1], reverse=True)[:limit]]
        finally:
            conn.close()

    # ── Bookmarks ───────────────────────────────────────────────────────────────

    def add_bookmark(self, bookmark: Bookmark) -> bool:
        """Add a bookmark. Returns False if duplicate."""
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO bookmarks (user_id, doc_id, skill, title, metadata, note)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    bookmark.user_id,
                    bookmark.doc_id,
                    bookmark.skill,
                    bookmark.title,
                    json.dumps(bookmark.metadata, ensure_ascii=False)
                    if bookmark.metadata
                    else None,
                    bookmark.note,
                ),
            )
            conn.commit()
            return True
        except Exception:  # UNIQUE constraint violation
            return False
        finally:
            conn.close()

    def remove_bookmark(self, doc_id: str) -> bool:
        conn = self._db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM bookmarks WHERE user_id = ? AND doc_id = ?",
                (self.user_id, doc_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def get_bookmarks(self, skill: str = None) -> list[Bookmark]:
        conn = self._db.get_connection()
        try:
            query = "SELECT * FROM bookmarks WHERE user_id = ?"
            params = [self.user_id]
            if skill:
                query += " AND skill = ?"
                params.append(skill)
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [
                Bookmark(
                    user_id=r["user_id"],
                    doc_id=r["doc_id"],
                    skill=r["skill"],
                    title=r["title"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                    note=r["note"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def search_bookmarks(self, query: str) -> list[Bookmark]:
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT * FROM bookmarks
                   WHERE user_id = ? AND (title LIKE ? OR note LIKE ?)
                   ORDER BY created_at DESC""",
                (self.user_id, f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [
                Bookmark(
                    user_id=r["user_id"],
                    doc_id=r["doc_id"],
                    skill=r["skill"],
                    title=r["title"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                    note=r["note"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ── Intent tracking ─────────────────────────────────────────────────────────

    def save_intent(self, topic: IntentTopic):
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO intent_tracking
                   (user_id, session_id, topic, confidence, queries)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    topic.user_id,
                    topic.session_id,
                    topic.topic,
                    topic.confidence,
                    json.dumps(topic.queries, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active_intents(self, max_age_days: int = 7) -> list[IntentTopic]:
        cutoff = (
            datetime.now() - timedelta(days=max_age_days)
        ).isoformat()
        conn = self._db.get_connection()
        try:
            rows = conn.execute(
                """SELECT * FROM intent_tracking
                   WHERE user_id = ? AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY created_at DESC""",
                (self.user_id, cutoff),
            ).fetchall()
            return [
                IntentTopic(
                    user_id=r["user_id"],
                    session_id=r["session_id"],
                    topic=r["topic"],
                    confidence=r["confidence"],
                    queries=json.loads(r["queries"]),
                )
                for r in rows
            ]
        finally:
            conn.close()
