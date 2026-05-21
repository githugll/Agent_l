"""Skill router: dispatch queries to TDoc or Spec retriever."""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class Skill(Enum):
    TDOC = "tdoc"
    SPEC = "spec"


class SkillRouter:
    """Routes queries to the appropriate retriever based on skill selection."""

    def __init__(self):
        self._tdoc_retriever = None
        self._spec_retriever = None

    def _init_tdoc_retriever(self):
        if self._tdoc_retriever is None:
            from retriever.tdocs.hybrid_retriever import HybridRetriever
            logger.info("Initializing TDoc retriever...")
            self._tdoc_retriever = HybridRetriever(device="cpu")
            logger.info("TDoc retriever ready")

    def _init_spec_retriever(self):
        if self._spec_retriever is None:
            from retriever.specs.spec_retriever import SpecRetriever
            logger.info("Initializing Spec retriever...")
            self._spec_retriever = SpecRetriever(device="cpu")
            logger.info("Spec retriever ready")

    def get_available_skills(self) -> list[Skill]:
        """Check which skills have data available."""
        skills = []
        tdoc_db = os.path.join("data", "tdocs", "chroma_db", "chroma.sqlite3")
        spec_db = os.path.join("data", "specs", "chroma_db", "chroma.sqlite3")
        if os.path.exists(tdoc_db):
            skills.append(Skill.TDOC)
        if os.path.exists(spec_db):
            skills.append(Skill.SPEC)
        return skills

    def retrieve(self, skill: Skill, query: str, **kwargs) -> list[dict]:
        """Route query to the appropriate retriever."""
        if skill == Skill.TDOC:
            self._init_tdoc_retriever()
            return self._tdoc_retriever.retrieve(query=query, **kwargs)
        elif skill == Skill.SPEC:
            self._init_spec_retriever()
            return self._spec_retriever.retrieve(query=query, **kwargs)
        else:
            raise ValueError(f"Unknown skill: {skill}")

    def get_retriever(self, skill: Skill):
        """Get the raw retriever instance for direct access."""
        if skill == Skill.TDOC:
            self._init_tdoc_retriever()
            return self._tdoc_retriever
        elif skill == Skill.SPEC:
            self._init_spec_retriever()
            return self._spec_retriever
