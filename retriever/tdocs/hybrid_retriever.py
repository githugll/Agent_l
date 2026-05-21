"""Hybrid retriever: dense + BM25 sparse + reranker + parent-doc context.

Features:
  - Timeout per stage (dense / BM25 / reranker)
  - Retry with backoff for Chroma connection
  - Minimum score threshold to filter low-quality hits
  - Detailed stage log for frontend display
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CHROMA_DIR = "data/tdocs/chroma_db"
CHUNKS_DIR = "data/tdocs/processed/chunks"
ALL_CHUNKS_FILE = CHUNKS_DIR + "/all_chunks.json"
GLOSSARY_FILE = "data/3gpp_glossary.json"

DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 2
DEFAULT_MIN_DENSE_SCORE = 0.05
DEFAULT_MIN_BM25_SCORE = 0.1


class HybridRetriever:
    """Hybrid retriever: dense + sparse + reranking + parent-doc context."""

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        chunks_file: str = ALL_CHUNKS_FILE,
        glossary_file: str = GLOSSARY_FILE,
        device: str = "cpu",
    ):
        logger.info("Initializing HybridRetriever...")

        # Load glossary
        with open(glossary_file, encoding="utf-8") as f:
            self.glossary = json.load(f)
        self.term_map = {k: v for k, v in self.glossary.items() if k != "_meta"}
        logger.info(f"Loaded {len(self.term_map)} glossary terms")

        # Load embedding model for query encoding (all-MiniLM-L6-v2 for CPU speed)
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        self.bge_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Embedding model loaded")

        # Connect to Chroma
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection("3gpp_tdocs")
        logger.info(f"Connected to Chroma: {self.collection.count()} chunks")

        # Load all chunks for BM25 and parent-doc lookup
        logger.info("Loading chunks for BM25 index...")
        with open(chunks_file, encoding="utf-8") as f:
            all_chunks = json.load(f)
        self.all_chunks = {c["chunk_id"]: c for c in all_chunks}

        # Build BM25 index on child chunks only
        child_chunks = [c for c in all_chunks if c["chunk_type"] == "child"]
        child_ids = [c["chunk_id"] for c in child_chunks]
        child_texts = [self._tokenize(c["text"]) for c in child_chunks]
        self.bm25 = BM25Okapi(child_texts)
        self.child_ids = child_ids
        logger.info(f"BM25 index built: {len(child_chunks)} child chunks")

        # Load parent chunks for context lookup
        parent_path = os.path.join(chroma_dir, "parent_chunks.json")
        if os.path.exists(parent_path):
            with open(parent_path, encoding="utf-8") as f:
                parent_chunks = json.load(f)
            self.parent_lookup = {c["chunk_id"]: c for c in parent_chunks}
            logger.info(f"Loaded {len(self.parent_lookup)} parent chunks")
        else:
            self.parent_lookup = {}
            logger.warning("No parent_chunks.json found")

        # Load reranker lazily (first use)
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading BGE-Reranker model...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            logger.info("Reranker loaded")
        return self._reranker

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenization for BM25."""
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9]{2,}", text)
        return tokens

    def _expand_query(self, query: str) -> str:
        """Expand query with 3GPP terminology full names."""
        parts = [query]
        query_upper = query.upper()
        for term, info in self.term_map.items():
            if term in query_upper or term.lower() in query.lower():
                full = info.get("full", "")
                if full and full.lower() not in query.lower():
                    parts.append(f"{term} {full}")
        return " ".join(parts)

    def _build_filter(self, working_group: str = None, companies: list = None) -> dict | None:
        """Build Chroma metadata filter."""
        conditions = []
        if working_group:
            conditions.append({"working_group": working_group})
        if companies:
            # Use $or to match any of the specified companies
            company_filters = [{"companies": co} for co in companies]
            if len(company_filters) == 1:
                conditions.append(company_filters[0])
            else:
                conditions.append({"$or": company_filters})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _merge_results(self, dense_ids: list, dense_scores: list, bm25_ids: list) -> list[dict]:
        """Merge and deduplicate results from dense and BM25 retrieval."""
        seen = set()
        merged = []
        # Interleave dense and BM25 results (dense first)
        for chunk_id, score in zip(dense_ids, dense_scores):
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append({"chunk_id": chunk_id, "dense_score": score, "bm25_score": 0.0})
        for chunk_id in bm25_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                merged.append({"chunk_id": chunk_id, "dense_score": 0.0, "bm25_score": 1.0})
        return merged

    def _attach_parent_context(self, results: list[dict]) -> list[dict]:
        """For each result, attach the parent chunk's full text for context."""
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

    def _timed(self, label: str, fn, *args, **kwargs):
        """Run fn with timeout and return (result, elapsed_ms, error)."""
        start = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            logger.info(f"[{label}] OK ({elapsed:.0f}ms)")
            return result, elapsed, None
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.warning(f"[{label}] FAILED after {elapsed:.0f}ms: {e}")
            return None, elapsed, str(e)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        working_group: str = None,
        companies: list = None,
        use_reranker: bool = True,
        use_glossary: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
        min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    ) -> list[dict]:
        """Retrieve relevant Tdoc chunks with timeout / retry / score filtering.

        Args:
            query: User query string
            top_k: Number of final results to return
            working_group: Filter by working group (e.g., "R2", "RAN1")
            companies: Filter by company names
            use_reranker: Whether to apply BGE-Reranker
            use_glossary: Whether to expand query with glossary
            timeout_s: Per-stage timeout in seconds (0 = no limit)
            max_retries: Max Chroma connection retries (0 = no retry)
            min_dense_score: Minimum dense similarity score to keep
            min_bm25_score: Minimum BM25 score to keep

        Returns:
            List of result dicts with chunk info and scores;
            empty list on complete failure.
        """
        stage_log = []  # [(stage_name, elapsed_ms, error_or_ok)]

        # ── Step 1: Query expansion ──────────────────────────────────────────────
        t0 = time.time()
        expanded_q = self._expand_query(query) if use_glossary else query
        expand_ms = (time.time() - t0) * 1000
        stage_log.append(f"术语扩展: {'无需扩展' if expanded_q == query else f'`{query}` → `{expanded_q}`'} | {expand_ms:.0f}ms")

        # ── Step 2: Dense retrieval (with retry) ────────────────────────────────
        t0 = time.time()
        chroma_filter = self._build_filter(working_group, companies)
        retrieve_k = top_k * 4

        dense_ids, dense_scores = [], []
        dense_err = None
        for attempt in range(max_retries + 1):
            _, elapsed, err = self._timed(
                "向量检索",
                lambda: self.collection.query(
                    query_embeddings=[self.bge_model.encode(
                        [expanded_q], normalize_embeddings=True
                    )[0].tolist()],
                    n_results=retrieve_k,
                    where=chroma_filter,
                    where_document=None,
                ),
            )
            if err is None:
                raw = _
                dense_ids = raw["ids"][0] if raw["ids"] else []
                dense_distances = raw["distances"][0] if raw["distances"] else []
                dense_scores = [1.0 / (d + 0.001) for d in dense_distances]
                # Apply minimum score threshold
                before = len(dense_ids)
                dense_ids, dense_scores = zip(*[
                    (i, s) for i, s in zip(dense_ids, dense_scores) if s >= min_dense_score
                ]) if dense_ids else ([], [])
                dense_ids, dense_scores = list(dense_ids), list(dense_scores)
                dropped = before - len(dense_ids)
                stage_log.append(
                    f"向量检索: {len(dense_ids)}条(过滤掉{before - len(dense_ids)}条, "
                    f"分数<{min_dense_score}) | {elapsed:.0f}ms"
                    if dropped else f"向量检索: {len(dense_ids)}条 | {elapsed:.0f}ms"
                )
                break
            else:
                dense_err = err
                if attempt < max_retries:
                    stage_log.append(f"向量检索 第{attempt+1}次失败, 重试... ({err})")
                else:
                    stage_log.append(f"向量检索 失败×{max_retries+1}: {err}")

        if dense_err and not dense_ids:
            stage_log.append("向量检索 完全失败")
            return []

        # ── Step 3: BM25 sparse retrieval ───────────────────────────────────────
        t0 = time.time()
        tokenized_q = self._tokenize(expanded_q)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ids_raw = [self.child_ids[i] for i in bm25_ranked]

        # Apply minimum BM25 score threshold
        raw_scores_map = {cid: bm25_scores[self.child_ids.index(cid)]
                          for cid in bm25_ids_raw if cid in self.child_ids}
        bm25_ids = []
        for cid in bm25_ids_raw:
            if cid not in raw_scores_map:
                continue
            if raw_scores_map[cid] >= min_bm25_score:
                bm25_ids.append(cid)
            if len(bm25_ids) >= retrieve_k:
                break

        bm25_elapsed = (time.time() - t0) * 1000
        dropped_bm25 = len(bm25_ids_raw) - len(bm25_ids)
        stage_log.append(
            f"BM25检索: {len(bm25_ids)}条(过滤掉{dropped_bm25}条, 分数<{min_bm25_score}) | "
            f"{bm25_elapsed:.0f}ms"
            if dropped_bm25 else f"BM25检索: {len(bm25_ids)}条 | {bm25_elapsed:.0f}ms"
        )

        # ── Step 4: Merge ─────────────────────────────────────────────────────
        merged = self._merge_results(dense_ids, dense_scores, bm25_ids)
        stage_log.append(f"合并: {len(merged)}条候选")

        if not merged:
            stage_log.append("合并后无候选结果")
            return []

        # ── Step 5: Reranking ─────────────────────────────────────────────────
        rerank_elapsed = 0
        if use_reranker and len(merged) >= 2:
            t0 = time.time()
            try:
                chunk_texts = [
                    self.all_chunks.get(r["chunk_id"], {}).get("text", "")[:512]
                    for r in merged
                ]
                pairs = [[query, text] for text in chunk_texts]
                rerank_scores = self.reranker.predict(pairs, batch_size=32)
                for r, score in zip(merged, rerank_scores):
                    r["rerank_score"] = float(score)
                merged.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                rerank_elapsed = (time.time() - t0) * 1000
                stage_log.append(f"Rerank: OK ({len(merged)}条重排) | {rerank_elapsed:.0f}ms")
            except Exception as e:
                stage_log.append(f"Rerank: 跳过({e})")

        # ── Step 6: Attach parent context + final results ──────────────────────
        final = merged[:top_k]
        self._attach_parent_context(final)

        # Enrich with metadata
        for r in final:
            chunk = self.all_chunks.get(r["chunk_id"], {})
            meta = chunk.get("metadata", {})
            r.update({
                "tdoc_number": meta.get("tdoc_number", ""),
                "working_group": meta.get("working_group", ""),
                "companies": meta.get("companies", ""),
                "meeting_id": meta.get("meeting_id", ""),
                "title": meta.get("title", ""),
                "doc_type": meta.get("doc_type", ""),
                "heading": chunk.get("heading", ""),
                "chunk_text": chunk.get("text", ""),
                "parent_text": r.get("parent_text", ""),
            })

        stage_log.append(f"最终返回: {len(final)}条")

        # Attach stage log for frontend
        final.append({"_stage_log": stage_log})

        total_ms = sum(s[1] for s in [
            (k, v) for k, v in [(s.split("|")[0], float(s.split("|")[-1].replace("ms","")))
                                   for s in stage_log if "|" in s]
        ])
        logger.info(f"检索完成 [{total_ms:.0f}ms]: {final[0].get('tdoc_number', '')} ...")

        return final

    def _filter_matches(self, chunk: dict, chroma_filter: dict) -> bool:
        """Check if a chunk matches a Chroma-style filter dict."""
        if "$and" in chroma_filter:
            return all(self._filter_matches(chunk, f) for f in chroma_filter["$and"])
        for key, val in chroma_filter.items():
            meta = chunk.get("metadata", {})
            chunk_val = meta.get(key, "")
            if isinstance(val, dict) and "$contains" in val:
                if val["$contains"] not in str(chunk_val):
                    return False
            elif str(val) != str(chunk_val):
                return False
        return True

    def retrieve_summary(self, results: list[dict]) -> str:
        """Format retrieval results as readable summary."""
        if not results:
            return "No results found."

        lines = [f"## 检索结果 ({len(results)} 条)\n"]
        seen_tdocs = set()
        for r in results:
            tdoc = r.get("tdoc_number", "Unknown")
            if tdoc in seen_tdocs:
                continue
            seen_tdocs.add(tdoc)
            wg = r.get("working_group", "")
            co = r.get("companies", "")
            title = r.get("title", r.get("heading", ""))[:80]
            score = r.get("rerank_score", r.get("dense_score", 0))
            lines.append(
                f"- **[{tdoc}]({wg})** | {co} | Score: {score:.3f}\n"
                f"  {title}"
            )
        return "\n".join(lines)


def test_retriever():
    """Quick test of the retriever."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ret = HybridRetriever(device="cpu")
    results = ret.retrieve(
        "beam management FR2",
        top_k=10,
        working_group="R2",
        use_reranker=True,
    )
    print(f"\nResults for 'beam management FR2': {len(results)}")
    for r in results[:5]:
        print(f"  {r.get('tdoc_number')}: {r.get('title','')[:60]} | score={r.get('rerank_score',0):.3f}")


if __name__ == "__main__":
    test_retriever()
