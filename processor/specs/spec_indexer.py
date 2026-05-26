"""Build Chroma vector database for 3GPP specifications."""

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_DIR = "data/specs/chroma_db"
CHUNKS_FILE = "data/specs/processed/chunks/all_spec_chunks.json"

INDEX_CHUNK_TYPES = ["child"]


def build_spec_vectorstore(
    chunks_file: str = CHUNKS_FILE,
    persist_dir: str = CHROMA_DIR,
    batch_size: int = 32,
) -> None:
    """Build Chroma vectorstore for specs.

    Indexes only child chunks; parent chunks saved separately for context lookup.
    """
    if not os.path.exists(chunks_file):
        logger.error(f"Chunks file not found: {chunks_file}")
        return

    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file) as f:
        all_chunks = json.load(f)

    child_chunks = [c for c in all_chunks if c["chunk_type"] in INDEX_CHUNK_TYPES]
    logger.info(
        f"Total chunks: {len(all_chunks)}, indexing {len(child_chunks)} child chunks"
    )

    if not child_chunks:
        logger.warning("No child chunks to index!")
        return

    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("Embedding model loaded")

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
    logger.info(f"Embeddings: shape={dense_embeddings.shape}")

    # Build Chroma collection
    import chromadb
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="3gpp_specs",
        metadata={"description": "3GPP Spec RAG vector store"},
    )

    BATCH_SIZE = 500
    total = len(child_chunks)

    for i in range(0, total, BATCH_SIZE):
        batch_chunks = child_chunks[i:i + BATCH_SIZE]
        batch_embs = dense_embeddings[i:i + BATCH_SIZE].tolist()
        batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

        for chunk, emb in zip(batch_chunks, batch_embs):
            meta = chunk.get("metadata", {})
            batch_ids.append(chunk["chunk_id"])
            batch_embeddings.append(emb)
            batch_documents.append(chunk["text"][:10000])
            batch_metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id", ""),
                "chunk_type": chunk["chunk_type"],
                "spec_number": meta.get("spec_number", ""),
                "section_path": meta.get("section_path", ""),
                "section_title": meta.get("section_title", "")[:200],
                "section_level": meta.get("section_level", 0),
                "release": meta.get("release", ""),
                "version": meta.get("version", ""),
                "char_count": len(chunk["text"]),
            })

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        logger.info(f"Indexed {min(i + BATCH_SIZE, total)}/{total}")

    logger.info(f"Vectorstore built: {collection.count()} chunks in '3gpp_specs'")

    # Save parent chunks for context lookup
    parent_chunks = [c for c in all_chunks if c["chunk_type"] == "parent"]
    parent_path = os.path.join(persist_dir, "parent_chunks.json")
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(parent_chunks, f, ensure_ascii=False)
    logger.info(f"Saved {len(parent_chunks)} parent chunks")

    return collection


def verify_spec_vectorstore(persist_dir: str = CHROMA_DIR) -> None:
    """Verify the spec vectorstore."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("3gpp_specs")
    count = collection.count()
    logger.info(f"Collection '3gpp_specs': {count} chunks")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    build_spec_vectorstore()
    verify_spec_vectorstore()
