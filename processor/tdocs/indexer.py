"""Build Chroma vector database with BGE-M3 embeddings."""

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_DIR = "data/tdocs/chroma_db"
CHUNKS_FILE = "data/tdocs/processed/chunks/all_chunks.json"

# Chunk types to index: only child chunks for retrieval
INDEX_CHUNK_TYPES = ["child"]


def build_vectorstore(
    chunks_file: str = CHUNKS_FILE,
    persist_dir: str = CHROMA_DIR,
    batch_size: int = 32,
) -> None:
    """Build Chroma vectorstore from chunk file.

    Only indexes child chunks (parent chunks stored separately for context lookup).
    """
    # Load chunks
    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file) as f:
        all_chunks = json.load(f)

    # Filter to child chunks only
    child_chunks = [c for c in all_chunks if c["chunk_type"] in INDEX_CHUNK_TYPES]
    logger.info(
        f"Total chunks: {len(all_chunks)}, indexing {len(child_chunks)} child chunks"
    )

    if not child_chunks:
        logger.warning("No child chunks to index!")
        return

    # Load embedding model (all-MiniLM-L6-v2 for CPU speed; swap to BAAI/bge-m3 if GPU available)
    logger.info("Loading embedding model (all-MiniLM-L6-v2, CPU optimized)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("Embedding model loaded")

    # Generate embeddings in batches
    texts = [c["text"] for c in child_chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    import numpy as np
    dense_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dense_embeddings = np.array(dense_embeddings)

    logger.info(f"Embeddings generated: shape={dense_embeddings.shape}")

    # Build Chroma collection
    import chromadb
    from chromadb.config import Settings

    # Clean up existing collection
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="3gpp_tdocs",
        metadata={"description": "3GPP Tdoc RAG vector store"},
    )

    # Add chunks in batches
    BATCH_SIZE = 500
    total = len(child_chunks)

    for i in range(0, total, BATCH_SIZE):
        batch_chunks = child_chunks[i : i + BATCH_SIZE]
        batch_embs = dense_embeddings[i : i + BATCH_SIZE].tolist()
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []

        for chunk, emb in zip(batch_chunks, batch_embs):
            meta = chunk["metadata"]
            companies_str = "|".join(meta.get("companies", [])) if isinstance(meta.get("companies"), list) else str(meta.get("companies", ""))

            batch_ids.append(chunk["chunk_id"])
            batch_embeddings.append(emb)
            batch_documents.append(chunk["text"][:10000])  # Truncate to avoid Chroma limit
            batch_metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id", ""),
                "chunk_type": chunk["chunk_type"],
                "tdoc_number": meta.get("tdoc_number", ""),
                "meeting_id": meta.get("meeting_id", ""),
                "working_group": meta.get("working_group", ""),
                "companies": companies_str,
                "meeting_number": int(meta.get("meeting_number", 0)),
                "doc_type": meta.get("doc_type", ""),
                "title": meta.get("title", "")[:200],
                "char_count": len(chunk["text"]),
            })

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        logger.info(f"Indexed {min(i + BATCH_SIZE, total)}/{total} chunks")

    logger.info(
        f"Vectorstore built: {collection.count()} chunks in collection '3gpp_tdocs'"
    )

    # Also save parent chunks to a separate JSON for context lookup
    parent_chunks = [c for c in all_chunks if c["chunk_type"] == "parent"]
    parent_path = os.path.join(persist_dir, "parent_chunks.json")
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(parent_chunks, f, ensure_ascii=False)
    logger.info(f"Saved {len(parent_chunks)} parent chunks to {parent_path}")

    return collection


def verify_vectorstore(persist_dir: str = CHROMA_DIR) -> None:
    """Verify the built vectorstore."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("3gpp_tdocs")

    count = collection.count()
    logger.info(f"Collection '3gpp_tdocs': {count} chunks")

    # Test query
    test_emb = [[0.0] * 1024]  # Placeholder
    try:
        results = collection.query(
            query_embeddings=test_emb,
            n_results=3,
            where={"chunk_type": "child"},
        )
        logger.info(f"Test query returned {len(results['documents'][0])} results")
    except Exception as e:
        logger.warning(f"Test query failed: {e}")

    # Check metadata filtering
    try:
        results = collection.get(
            where={"working_group": "R2"},
            limit=5,
        )
        logger.info(f"Metadata filter (R2): {len(results['documents'])} results")
    except Exception as e:
        logger.warning(f"Metadata filter test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    build_vectorstore()
    verify_vectorstore()
