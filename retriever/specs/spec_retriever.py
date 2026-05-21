"""Spec retriever: hybrid dense + BM25 + rerank for 3GPP specifications.

Features:
  - Section reference detection (e.g. "38.321 5.1" → targeted retrieval)
  - Spec/section metadata filtering
  - Spec-aware result formatting (TS 38.321 Section 5.1)
  - Glossary-based query expansion with _spec_sections mapping
"""

import json
import logging
import os
import re
import time

import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CHROMA_DIR = "data/specs/chroma_db"
CHUNKS_DIR = "data/specs/processed/chunks"
ALL_CHUNKS_FILE = CHUNKS_DIR + "/all_spec_chunks.json"
GLOSSARY_FILE = "data/3gpp_glossary.json"

DEFAULT_TIMEOUT_S = 30
DEFAULT_MIN_DENSE_SCORE = 0.05
DEFAULT_MIN_BM25_SCORE = 0.1

# Detect section references: "38.321 5.1", "TS 38.321 section 5.1", "38321-5.1"
SECTION_REF_RE = re.compile(
    r"(?:TS\s+)?(\d{2}\.?\d{2,3})"
    r"(?:\s*[-–]?\s*|Section\s+|Sec\.?\s+|section\s+|\s+)"
    r"(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


class SpecRetriever:
    """Hybrid retriever for 3GPP specifications."""

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        chunks_file: str = ALL_CHUNKS_FILE,
        glossary_file: str = GLOSSARY_FILE,
        device: str = "cpu",
    ):
        logger.info("Initializing SpecRetriever...")

        # Load glossary
        with open(glossary_file, encoding="utf-8") as f:
            self.glossary = json.load(f)
        self.term_map = {k: v for k, v in self.glossary.items() if k not in ("_meta", "_spec_sections")}
        self.spec_sections = self.glossary.get("_spec_sections", {})
        logger.info(f"Loaded {len(self.term_map)} terms, {len(self.spec_sections)} spec sections")

        # Load embedding model
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        self.bge_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Embedding model loaded")

        # Connect to Chroma
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection("3gpp_specs")
        logger.info(f"Connected to Chroma: {self.collection.count()} chunks")

        # Load chunks for BM25 and parent lookup
        logger.info("Loading chunks for BM25 index...")
        with open(chunks_file, encoding="utf-8") as f:
            all_chunks = json.load(f)
        self.all_chunks = {c["chunk_id"]: c for c in all_chunks}

        child_chunks = [c for c in all_chunks if c["chunk_type"] == "child"]
        child_ids = [c["chunk_id"] for c in child_chunks]
        child_texts = [self._tokenize(c["text"]) for c in child_chunks]
        self.bm25 = BM25Okapi(child_texts)
        self.child_ids = child_ids
        logger.info(f"BM25 index built: {len(child_chunks)} child chunks")

        # Parent chunk lookup
        parent_path = os.path.join(chroma_dir, "parent_chunks.json")
        if os.path.exists(parent_path):
            with open(parent_path, encoding="utf-8") as f:
                parent_chunks = json.load(f)
            self.parent_lookup = {c["chunk_id"]: c for c in parent_chunks}
            logger.info(f"Loaded {len(self.parent_lookup)} parent chunks")
        else:
            self.parent_lookup = {}

        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading BGE-Reranker...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            logger.info("Reranker loaded")
        return self._reranker

    # ── Query parsing ─────────────────────────────────────────────────────────

    def _parse_section_ref(self, query: str) -> dict | None:
        """Detect spec section references in query. Returns {spec_number, section_path} or None."""
        match = SECTION_REF_RE.search(query)
        if not match:
            return None
        spec_raw = match.group(1)
        section_path = match.group(2)
        # Normalize spec number: "38321" → "38.321" or "38.321" → "38.321"
        if len(spec_raw) == 5:
            spec_number = f"{spec_raw[:2]}.{spec_raw[2:]}"
        else:
            spec_number = spec_raw
        return {"spec_number": spec_number, "section_path": section_path}

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"[a-zA-Z0-9]{2,}", text)

    def _expand_query(self, query: str) -> str:
        """Expand query with glossary terms and spec section mappings."""
        parts = [query]
        query_lower = query.lower()

        # Standard term expansion
        for term, info in self.term_map.items():
            if term in query_lower:
                full = info.get("full", "")
                if full and full.lower() not in query_lower:
                    parts.append(f"{term} {full}")

        # Spec section expansion
        for term, mapping in self.spec_sections.items():
            if term in query_lower:
                spec = mapping.get("spec", "")
                section = mapping.get("section", "")
                if spec and section:
                    parts.append(f"TS {spec} Section {section}")

        return " ".join(parts)

    # ── Filtering ────────────────────────────────────────────────────────────

    def _build_filter(self, spec_number: str = None, section_prefix: str = None) -> dict | None:
        """Build Chroma metadata filter for spec_number.

        Note: ChromaDB where clause does NOT support regex.
        Section prefix filtering is done as post-filter in retrieve().
        """
        if spec_number:
            return {"spec_number": spec_number}
        return None

    def _attach_parent_context(self, results: list[dict]) -> list[dict]:
        for r in results:
            parent_id = r.get("parent_id", "")
            if parent_id and parent_id in self.parent_lookup:
                parent = self.parent_lookup[parent_id]
                r["parent_text"] = parent["text"]
                r["parent_heading"] = parent.get("heading", "")
            else:
                r["parent_text"] = ""
                r["parent_heading"] = ""
        return results

    def _merge_results(self, dense_ids, dense_scores, bm25_ids) -> list[dict]:
        seen = set()
        merged = []
        for chunk_id, score in zip(dense_ids, dense_scores):
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append({"chunk_id": chunk_id, "dense_score": score, "bm25_score": 0.0})
        for chunk_id in bm25_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append({"chunk_id": chunk_id, "dense_score": 0.0, "bm25_score": 1.0})
        return merged

    # ── Main retrieval ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        spec_filter: str = None,
        use_reranker: bool = True,
        use_glossary: bool = True,
    ) -> list[dict]:
        """Retrieve relevant spec chunks.

        Args:
            query: User query (may include section references like "38.321 5.1")
            top_k: Number of final results
            spec_filter: Restrict to a specific spec number (e.g. "38.321")
            use_reranker: Apply BGE reranker
            use_glossary: Apply query expansion

        Returns:
            List of result dicts with spec_number, section_path, chunk_text, scores.
        """
        stage_log = []

        # Step 1: Detect section reference
        section_ref = self._parse_section_ref(query)
        if section_ref:
            stage_log.append(
                f"章节检测: '{section_ref['spec_number']} Section {section_ref['section_path']}'"
            )
            # Use section ref as spec filter if not explicitly provided
            if not spec_filter:
                spec_filter = section_ref["spec_number"]

        # Step 2: Query expansion
        expanded_q = self._expand_query(query) if use_glossary else query
        if expanded_q != query:
            stage_log.append(f"术语扩展: `{query}` → `{expanded_q}`")
        else:
            stage_log.append("术语扩展: 无需扩展")

        # Step 3: Dense retrieval
        section_prefix = section_ref["section_path"] if section_ref else None
        chroma_filter = self._build_filter(spec_filter)
        # Use larger retrieve_k when section filtering to ensure we get enough candidates
        retrieve_k = top_k * 10 if section_prefix else top_k * 4

        dense_ids, dense_scores = [], []
        try:
            query_emb = self.bge_model.encode([expanded_q], normalize_embeddings=True)[0].tolist()
            raw = self.collection.query(
                query_embeddings=[query_emb],
                n_results=retrieve_k,
                where=chroma_filter,
            )
            dense_ids = raw["ids"][0] if raw["ids"] else []
            dense_distances = raw["distances"][0] if raw["distances"] else []
            dense_scores = [1.0 / (d + 0.001) for d in dense_distances]
            # Apply score threshold
            pairs = list(zip(dense_ids, dense_scores))
            pairs = [(i, s) for i, s in pairs if s >= DEFAULT_MIN_DENSE_SCORE]
            dense_ids, dense_scores = zip(*pairs) if pairs else ([], [])
            dense_ids, dense_scores = list(dense_ids), list(dense_scores)
            stage_log.append(f"向量检索: {len(dense_ids)}条 | chroma filter={chroma_filter is not None}")
        except Exception as e:
            stage_log.append(f"向量检索失败: {e}")
            return []

        # Step 4: BM25
        tokenized_q = self._tokenize(expanded_q)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ids = []
        for idx in ranked:
            cid = self.child_ids[idx]
            score = bm25_scores[idx]
            if score >= DEFAULT_MIN_BM25_SCORE:
                bm25_ids.append(cid)
            if len(bm25_ids) >= retrieve_k:
                break
        stage_log.append(f"BM25检索: {len(bm25_ids)}条")

        # Step 5: Merge
        merged = self._merge_results(dense_ids, dense_scores, bm25_ids)
        stage_log.append(f"合并: {len(merged)}条候选")

        if not merged:
            return []

        # Step 5b: Post-filter by section prefix (ChromaDB doesn't support regex in where)
        if section_prefix:
            before = len(merged)
            merged = [
                r for r in merged
                if r.get("chunk_id") in self.all_chunks
                and self.all_chunks[r["chunk_id"]].get("metadata", {}).get("section_path", "").startswith(section_prefix)
            ]
            stage_log.append(f"章节过滤({section_prefix}): {before}→{len(merged)}条")

        # Step 6: Rerank
        if use_reranker and len(merged) >= 2:
            try:
                chunk_texts = [
                    self.all_chunks.get(r["chunk_id"], {}).get("text", "")[:512]
                    for r in merged
                ]
                pairs = [[query, t] for t in chunk_texts]
                rerank_scores = self.reranker.predict(pairs, batch_size=32)
                for r, score in zip(merged, rerank_scores):
                    r["rerank_score"] = float(score)
                merged.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                stage_log.append(f"Rerank: OK ({len(merged)}条重排)")
            except Exception as e:
                stage_log.append(f"Rerank: 跳过({e})")

        # Step 7: Attach context + final results
        final = merged[:top_k]
        self._attach_parent_context(final)

        # Enrich with metadata
        for r in final:
            chunk = self.all_chunks.get(r["chunk_id"], {})
            meta = chunk.get("metadata", {})
            r.update({
                "spec_number": meta.get("spec_number", ""),
                "section_path": meta.get("section_path", ""),
                "section_title": meta.get("section_title", ""),
                "section_level": meta.get("section_level", 0),
                "heading": chunk.get("heading", ""),
                "chunk_text": chunk.get("text", ""),
                "parent_text": r.get("parent_text", ""),
            })

        stage_log.append(f"最终返回: {len(final)}条")
        final.append({"_stage_log": stage_log})

        logger.info(f"Spec retrieval [{query[:30]}]: {len(final) - 1} results")
        return final

    def retrieve_summary(self, results: list[dict]) -> str:
        """Format retrieval results as readable summary."""
        if not results:
            return "No results found."

        lines = [f"## Spec 检索结果 ({len(results) - 1} 条)\n"]
        seen = set()
        for r in results:
            if "_stage_log" in r:
                continue
            spec = r.get("spec_number", "")
            sec = r.get("section_path", "")
            if not spec:
                continue
            key = f"{spec}:{sec}"
            if key in seen:
                continue
            seen.add(key)
            title = r.get("section_title", r.get("heading", ""))[:80]
            score = r.get("rerank_score", r.get("dense_score", 0))
            lines.append(
                f"- **TS {spec} §{sec}** — {title} "
                f"| score={score:.3f}"
            )
        return "\n".join(lines)
