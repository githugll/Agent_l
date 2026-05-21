"""3GPP Agent Memory Module.

Usage in app.py:
    from memory import MemoryManager

    # In on_chat_start:
    mem = MemoryManager(session_id=session_id, user_id=user_id)
    cl.user_session.set("memory", mem)

    # In on_message (after retrieval):
    mem: MemoryManager = cl.user_session.get("memory")
    mem.record_query(query, skill, results, top_k, filters)

    # In analysis:
    context = mem.build_context(skill)
    # → injected into system prompt
"""

from memory.db import MemoryDB
from memory.models import (
    Bookmark,
    ConversationTurn,
    IntentTopic,
    RetrievalSnapshot,
    SearchHistoryEntry,
    UserPreferences,
)
from memory.long_term import UserMemory
from memory.short_term import ConversationMemory
from memory.context_builder import PromptContextBuilder


class MemoryManager:
    """Facade that owns all memory subsystems for one session.

    One instance per Chainlit session, stored in cl.user_session["memory"].
    """

    def __init__(
        self,
        session_id: str,
        user_id: str = "default",
        db: MemoryDB = None,
    ):
        self._db = db or MemoryDB.get_instance()
        self.conversation = ConversationMemory(session_id, self._db)
        self.user_memory = UserMemory(user_id, self._db)
        self.preferences: UserPreferences = self.user_memory.load_preferences()
        self.context_builder = PromptContextBuilder(
            self.conversation, self.user_memory, self.preferences
        )
        # Trigger lazy cleanup on session start
        self._db.purge_expired()

    def record_query(
        self,
        query: str,
        skill: str,
        results: list[dict] = None,
        top_k: int = 20,
        filters: dict = None,
    ):
        """Convenience: record a user query + optional retrieval in one call."""
        turn = self.conversation.add_user_query(query, skill, query_type="search")
        if results is not None:
            self.conversation.add_retrieval(query, skill, results, top_k, filters)
            self.user_memory.record_search(
                SearchHistoryEntry(
                    user_id=self.user_memory.user_id,
                    session_id=self.conversation.session_id,
                    query=query,
                    skill=skill,
                    filters=filters,
                    result_count=len(results),
                )
            )

    def record_analysis(self, analysis_text: str, skill: str):
        """Record an AI analysis response."""
        self.conversation.add_assistant_response(
            analysis_text, skill, query_type="analysis"
        )

    def build_context(self, skill: str) -> str:
        """Build the full prompt context for the current turn."""
        return self.context_builder.build_system_context(skill)

    def build_retrieval_context(
        self, results: list[dict], query: str
    ) -> str | None:
        """Build retrieval-specific context for follow-up queries."""
        return self.context_builder.build_retrieval_context(results, query)

    def save_preferences_from_settings(self, settings: dict, skill: str):
        """Persist Chainlit ChatSettings as user preferences."""
        import json

        self.preferences.preferred_skill = skill
        if "top_k" in settings:
            self.preferences.top_k = int(settings["top_k"])
        if "working_group" in settings:
            wg = settings["working_group"]
            if wg and wg not in self.preferences.working_groups:
                self.preferences.working_groups = (
                    [wg] + [w for w in self.preferences.working_groups if w != wg]
                )[:5]
        if "company" in settings:
            co = settings["company"]
            if co and co not in self.preferences.companies:
                self.preferences.companies = (
                    [co] + [c for c in self.preferences.companies if c != co]
                )[:5]
        if "spec_filter" in settings:
            pass  # spec_filter not persisted as a long-term preference yet
        self.user_memory.save_preferences(self.preferences)

    def add_bookmark(
        self,
        doc_id: str,
        skill: str,
        title: str = None,
        note: str = None,
    ) -> bool:
        return self.user_memory.add_bookmark(
            Bookmark(
                user_id=self.user_memory.user_id,
                doc_id=doc_id,
                skill=skill,
                title=title,
                note=note,
            )
        )

    def get_bookmarks(self, skill: str = None) -> list[Bookmark]:
        return self.user_memory.get_bookmarks(skill)

    def search_bookmarks(self, query: str) -> list[Bookmark]:
        return self.user_memory.search_bookmarks(query)
