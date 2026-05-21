"""Assembles memory data into LLM-consumable prompt context."""

import json
import logging
import re
from typing import Optional

from .models import UserPreferences
from .short_term import ConversationMemory
from .long_term import UserMemory

logger = logging.getLogger(__name__)

# Load glossary for keyword highlighting
_GLOSSARY: dict = {}
_GLOSSARY_TERMS: set[str] = set()
try:
    with open("data/3gpp_glossary.json", encoding="utf-8") as f:
        _GLOSSARY = json.load(f)
    if isinstance(_GLOSSARY, dict):
        _GLOSSARY_TERMS = {k.lower() for k in _GLOSSARY.keys() if not k.startswith("_")}
    elif isinstance(_GLOSSARY, list):
        _GLOSSARY_TERMS = {e.get("term", "").lower() for e in _GLOSSARY if e.get("term")}
except Exception:
    pass


class PromptContextBuilder:
    """Combines short-term + long-term memory into structured prompt text.

    Single point where memory data is assembled into text prepended to the
    system prompt or injected as context messages for the LLM.
    """

    def __init__(
        self,
        conversation: ConversationMemory,
        user_memory: UserMemory,
        preferences: UserPreferences,
    ):
        self.conversation = conversation
        self.user_memory = user_memory
        self.preferences = preferences

    def build_system_context(self, skill: str) -> str:
        """Build the contextual section to append to the system prompt.

        Structure:
          [用户偏好]
          [当前研究方向]
          [对话上下文]
        """
        parts: list[str] = []

        # 1. User preferences
        prefs_parts = self._build_preferences_section()
        if prefs_parts:
            parts.append(prefs_parts)

        # 2. Active research direction
        intent_parts = self._build_intent_section()
        if intent_parts:
            parts.append(intent_parts)

        # 3. Conversation summary
        ctx_parts = self._build_conversation_section(skill)
        if ctx_parts:
            parts.append(ctx_parts)

        return "\n\n".join(parts)

    def build_retrieval_context(
        self, current_results: list[dict], query: str
    ) -> Optional[str]:
        """Build context about how current results relate to previous ones."""
        prev = self.conversation.get_retrieval_association(query)
        if not prev:
            return None

        prev_query = prev.query
        prev_count = prev.result_count
        skill = prev.skill

        lines = [
            f"[关联历史检索]",
            f"上一轮查询「{prev_query}」（{skill.upper()}，{prev_count} 条结果）",
            f"当前查询「{query}」是上述检索的跟进。",
            "相关文档：",
        ]
        for r in prev.top_results[:5]:
            lines.append(f"  - [{r['doc_id']}] {r['title']} (相关度 {r['score']})")

        return "\n".join(lines)

    def build_preferences_hint(self) -> str:
        """Build a short hint about user preferences for the system prompt."""
        prefs = self.preferences
        parts = []

        if prefs.working_groups:
            wgs = ", ".join(prefs.working_groups[:3])
            parts.append(f"用户偏好工作组: {wgs}")

        if prefs.companies:
            cos = ", ".join(prefs.companies[:3])
            parts.append(f"用户偏好公司: {cos}")

        if parts:
            return "用户偏好：" + "；".join(parts)
        return ""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_preferences_section(self) -> str:
        prefs = self.preferences
        parts = ["[用户偏好]"]

        if prefs.preferred_skill:
            parts.append(f"- 常用 Skill: {prefs.preferred_skill.upper()}")

        if prefs.working_groups:
            wgs = ", ".join(prefs.working_groups[:4])
            parts.append(f"- 常查工作组: {wgs}")

        if prefs.companies:
            cos = ", ".join(prefs.companies[:4])
            parts.append(f"- 关注公司: {cos}")

        if prefs.top_k and prefs.top_k != 20:
            parts.append(f"- 默认结果数: {prefs.top_k}")

        # Top topics from search history
        topic_freq = self.user_memory.get_topic_frequency(days=30)
        if topic_freq:
            top_topics = list(topic_freq.items())[:5]
            topic_str = "、".join(f"{t}({c}次)" for t, c in top_topics)
            parts.append(f"- 高频搜索主题: {topic_str}")

        lines = "\n".join(parts)
        return lines if len(lines) > 20 else ""

    def _build_intent_section(self) -> str:
        # From current session
        session_intent = self.conversation.infer_intent()
        # From long-term
        active_intents = self.user_memory.get_active_intents(max_age_days=7)
        long_term_topics = [it.topic for it in active_intents[:5]]

        # Also get topic frequency
        topic_freq = self.user_memory.get_topic_frequency(days=7)
        trending = list(topic_freq.items())[:5]

        all_topics = list(
            dict.fromkeys(
                [t for t in session_intent if t in _GLOSSARY_TERMS]
                + long_term_topics
                + [t for t, _ in trending]
            )
        )

        if not all_topics:
            return ""

        # Highlight glossary terms (look up original casing)
        glossary_highlighted = []
        for t in all_topics[:6]:
            if isinstance(_GLOSSARY, dict):
                for key in _GLOSSARY:
                    if key.lower() == t and not key.startswith("_"):
                        glossary_highlighted.append(key)
                        break
                else:
                    glossary_highlighted.append(t)
            else:
                glossary_highlighted.append(t)

        return (
            "[当前研究方向]\n"
            f"用户正在研究：{'、'.join(glossary_highlighted)}"
        )

    def _build_conversation_section(self, skill: str) -> str:
        recent = self.conversation.get_recent_context(max_turns=6)
        if not recent:
            return ""

        lines = ["[对话上下文]"]
        token_count = 0
        max_tokens = 350

        for turn in recent:
            if turn.role == "user":
                content = turn.content[:120]
                extra = ""
                if turn.query_type == "search":
                    retrieval = self.conversation.get_last_retrieval()
                    if retrieval and retrieval.skill == skill:
                        extra = f" → {retrieval.result_count} 条结果"
                line = f"- 用户: 「{content}」{extra}"
            else:
                content = turn.content[:80]
                line = f"- AI: 「{content}」"

            token_count += len(line) // 2
            if token_count > max_tokens:
                break
            lines.append(line)

        return "\n".join(lines) if len(lines) > 1 else ""
