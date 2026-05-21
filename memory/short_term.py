"""Short-term conversation memory (in-session context)."""

import json
import logging
import re
from typing import Optional

from .db import MemoryDB
from .models import ConversationTurn, RetrievalSnapshot

logger = logging.getLogger(__name__)

# Load 3GPP glossary for intent extraction
_GLOSSARY_KEYWORDS: set[str] = set()
try:
    with open("data/3gpp_glossary.json", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        _GLOSSARY_KEYWORDS = {k.lower() for k in raw.keys() if not k.startswith("_")}
    elif isinstance(raw, list):
        _GLOSSARY_KEYWORDS = {e.get("term", "").lower() for e in raw if e.get("term")}
except Exception:
    pass

_RESULT_REF_RE = re.compile(
    r"(?:第|result|result|#)(\d+)",
    re.IGNORECASE,
)


class ConversationMemory:
    """In-session conversation context manager.

    Stores turn-by-turn data in memory (fast access) and persists to SQLite
    (recovery / analytics).
    """

    def __init__(self, session_id: str, db: MemoryDB):
        self.session_id = session_id
        self._db = db
        self._turns: list[ConversationTurn] = []
        self._current_turn_index: int = 0
        self._last_retrieval: Optional[RetrievalSnapshot] = None
        self._intent_buffer: list[str] = []  # recent query keywords

    def add_user_query(
        self, content: str, skill: str, query_type: str = "search"
    ) -> int:
        """Record a user message. Returns the turn_index."""
        self._intent_buffer.append(content.lower())
        if len(self._intent_buffer) > 10:
            self._intent_buffer = self._intent_buffer[-10:]

        turn = ConversationTurn(
            session_id=self.session_id,
            turn_index=self._current_turn_index,
            role="user",
            content=content,
            skill=skill,
            query_type=query_type,
        )
        self._turns.append(turn)
        self._current_turn_index += 1
        self._persist_turn(turn)
        return turn.turn_index

    def add_assistant_response(
        self, content: str, skill: str, query_type: str = "search"
    ):
        """Record an assistant response."""
        turn = ConversationTurn(
            session_id=self.session_id,
            turn_index=self._current_turn_index,
            role="assistant",
            content=content,
            skill=skill,
            query_type=query_type,
        )
        self._turns.append(turn)
        self._current_turn_index += 1
        self._persist_turn(turn)

    def add_retrieval(
        self,
        query: str,
        skill: str,
        results: list[dict],
        top_k: int = 20,
        filters: dict = None,
    ):
        """Store a retrieval snapshot and associate it with the current turn."""
        # Build abbreviated result summary
        top_results = []
        for r in results[:10]:
            if skill == "tdoc":
                top_results.append(
                    {
                        "doc_id": r.get("tdoc_number", ""),
                        "title": (r.get("title") or r.get("heading") or "")[:80],
                        "score": round(
                            r.get("rerank_score") or r.get("dense_score", 0), 3
                        ),
                    }
                )
            else:
                top_results.append(
                    {
                        "doc_id": f"TS {r.get('spec_number', '')}:{r.get('section_path', '')}",
                        "title": (r.get("section_title") or r.get("heading") or "")[:80],
                        "score": round(
                            r.get("rerank_score") or r.get("dense_score", 0), 3
                        ),
                    }
                )

        snapshot = RetrievalSnapshot(
            session_id=self.session_id,
            turn_index=self._current_turn_index - 1,  # associate with user turn
            query=query,
            skill=skill,
            top_k=top_k,
            filters=filters,
            result_count=len(results),
            top_results=top_results,
        )
        self._last_retrieval = snapshot
        self._persist_snapshot(snapshot)

    def get_recent_context(self, max_turns: int = 6) -> list[ConversationTurn]:
        """Return the last N conversation turns for prompt context."""
        return self._turns[-max_turns:] if self._turns else []

    def get_last_retrieval(self) -> Optional[RetrievalSnapshot]:
        return self._last_retrieval

    def get_retrieval_association(self, query: str) -> Optional[RetrievalSnapshot]:
        """Find a previous retrieval relevant to the given follow-up query."""
        if not self._last_retrieval:
            return None

        q_lower = query.lower()
        # Check if query references a result number
        m = _RESULT_REF_RE.search(q_lower)
        if m:
            return self._last_retrieval

        # Check for follow-up patterns
        follow_up_patterns = ["more", "还有", "另外", "其他", "tell me", "详情", "详细"]
        if any(p in q_lower for p in follow_up_patterns):
            return self._last_retrieval

        # Check keyword overlap with last retrieval
        last_query_words = set(self._last_retrieval.query.lower().split())
        query_words = set(q_lower.split())
        overlap = last_query_words & query_words
        if len(overlap) >= 2 and overlap != last_query_words:
            return self._last_retrieval

        return None

    def infer_intent(self) -> list[str]:
        """Analyze recent queries to extract research direction keywords."""
        recent = self._intent_buffer[-5:]
        if not recent:
            return []

        all_words: list[str] = []
        for q in recent:
            # Extract words longer than 3 chars, skipping stopwords
            words = re.findall(r"[a-zA-Z0-9一-鿿]{3,}", q)
            all_words.extend(w.lower() for w in words)

        # Count frequency
        freq: dict[str, int] = {}
        for w in all_words:
            freq[w] = freq.get(w, 0) + 1

        # Also include glossary terms found in queries
        glossary_found = [w for w in all_words if w in _GLOSSARY_KEYWORDS]

        # Return top keywords + glossary terms
        top_keywords = sorted(freq, key=freq.get, reverse=True)[:8]
        combined = list(dict.fromkeys(glossary_found + top_keywords))
        return combined[:6]

    def get_context_summary(self, max_tokens: int = 500) -> str:
        """Generate a compact text summary of the conversation so far."""
        recent = self.get_recent_context(max_turns=6)
        if not recent:
            return ""

        lines = ["[对话上下文]"]
        tokens_so_far = 0
        for turn in recent:
            prefix = "用户" if turn.role == "user" else "AI"
            content = turn.content[:150]
            suffix = (
                f" | 检索{turn.result_summary or ''}"
                if turn.query_type == "search" and turn.role == "user"
                else ""
            )
            line = f"- {prefix}: {content}{suffix}"
            tokens_so_far += len(line) // 2
            if tokens_so_far > max_tokens:
                break
            lines.append(line)

        return "\n".join(lines)

    def clear(self):
        """Reset in-memory buffers (called on skill switch)."""
        self._turns.clear()
        self._current_turn_index = 0
        self._last_retrieval = None
        self._intent_buffer.clear()

    def _persist_turn(self, turn: ConversationTurn):
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO conversation_turns
                   (session_id, turn_index, role, content, skill, query_type,
                    result_summary, intent_tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn.session_id,
                    turn.turn_index,
                    turn.role,
                    turn.content,
                    turn.skill,
                    turn.query_type,
                    json.dumps(turn.result_summary, ensure_ascii=False)
                    if turn.result_summary
                    else None,
                    json.dumps(turn.intent_tags, ensure_ascii=False)
                    if turn.intent_tags
                    else None,
                    turn.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_snapshot(self, snapshot: RetrievalSnapshot):
        conn = self._db.get_connection()
        try:
            conn.execute(
                """INSERT INTO retrieval_snapshots
                   (session_id, turn_index, query, skill, top_k, filters,
                    result_count, top_results, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.session_id,
                    snapshot.turn_index,
                    snapshot.query,
                    snapshot.skill,
                    snapshot.top_k,
                    json.dumps(snapshot.filters, ensure_ascii=False)
                    if snapshot.filters
                    else None,
                    snapshot.result_count,
                    json.dumps(snapshot.top_results, ensure_ascii=False),
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
