"""Dataclass models for memory module."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationTurn:
    session_id: str
    turn_index: int
    role: str  # "user" | "assistant" | "system"
    content: str
    skill: Optional[str] = None
    query_type: Optional[str] = None  # "search" | "analysis" | "bookmark" | "settings_change"
    result_summary: Optional[dict] = None
    intent_tags: Optional[list[str]] = None
    created_at: Optional[str] = None


@dataclass
class RetrievalSnapshot:
    session_id: str
    turn_index: int
    query: str
    skill: str
    top_k: int = 20
    filters: Optional[dict] = None
    result_count: int = 0
    top_results: list[dict] = field(default_factory=list)


@dataclass
class UserPreferences:
    user_id: str
    preferred_skill: str = "tdoc"
    working_groups: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    top_k: int = 20
    language: str = "zh-CN"


@dataclass
class Bookmark:
    user_id: str
    doc_id: str
    skill: str
    title: Optional[str] = None
    metadata: Optional[dict] = None
    note: Optional[str] = None


@dataclass
class SearchHistoryEntry:
    user_id: str
    session_id: str
    query: str
    skill: str
    filters: Optional[dict] = None
    result_count: int = 0
    clicked_doc_ids: Optional[list[str]] = None


@dataclass
class IntentTopic:
    user_id: str
    session_id: str
    topic: str
    confidence: float = 0.0
    queries: list[str] = field(default_factory=list)
