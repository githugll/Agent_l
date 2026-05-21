"""Hierarchical chunking for 3GPP Tdoc documents.

Strategy:
  Parent chunk: one per Tdoc section (by numbered headings like 1, 1.1, 2.3.1)
  Child chunk: ~512 tokens with 64-token overlap, linked to parent via parent_id
"""

import hashlib
import json
import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Section header pattern: "1.", "1.1", "2.3.1", "3.1.2.1" at the start of a line
SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+){0,3})\s+(.{3,80})",
    re.MULTILINE,
)

# Approximate token count: ~4 chars per token for English, ~2 for Chinese
CHARS_PER_TOKEN = 3.5

CHILD_MAX_CHARS = int(512 * CHARS_PER_TOKEN)  # ~1792 chars
CHILD_OVERLAP_CHARS = int(64 * CHARS_PER_TOKEN)  # ~224 chars


def _split_into_sections(text: str) -> list[dict]:
    """Split text into sections based on numbered headings."""
    sections = []
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        # No section headers found - treat entire text as one section
        return [{"heading": "Full Document", "text": text}]

    # Add content before first section as "Header" section
    if matches[0].start() > 100:
        header_text = text[:matches[0].start()].strip()
        if header_text:
            sections.append({"heading": "Header", "text": header_text})

    for i, match in enumerate(matches):
        section_num = match.group(1)
        heading = match.group(2).strip()
        start = match.start()

        # End at next section start or end of text
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        if section_text:
            sections.append({
                "heading": f"{section_num} {heading}",
                "text": section_text,
            })

    return sections


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Split into paragraphs
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not para.strip():
            continue

        if len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of current chunk
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(tdoc_number: str, text: str, metadata: dict) -> list[dict]:
    """Create hierarchical chunks for a single Tdoc.

    Returns list of chunks, each with:
      chunk_id, parent_id, chunk_type ("parent"|"child"),
      text, metadata
    """
    chunks = []
    sections = _split_into_sections(text)

    # Use a global counter within this document to ensure unique IDs
    section_counter = 0

    for section in sections:
        # Include section_counter + text hash for uniqueness
        text_hash = hashlib.md5(section["text"][:200].encode()).hexdigest()[:8]
        parent_id = f"{tdoc_number}_p{section_counter}_{text_hash}"

        # Create parent chunk
        parent_chunk = {
            "chunk_id": parent_id,
            "parent_id": "",
            "chunk_type": "parent",
            "text": section["text"],
            "heading": section["heading"],
            "metadata": {**metadata, "tdoc_number": tdoc_number},
        }
        chunks.append(parent_chunk)

        # Create child chunks from the section
        if len(section["text"]) > CHILD_MAX_CHARS:
            child_texts = _chunk_text(section["text"], CHILD_MAX_CHARS, CHILD_OVERLAP_CHARS)
            for j, child_text in enumerate(child_texts):
                child_hash = hashlib.md5(child_text[:200].encode()).hexdigest()[:8]
                child_id = f"{tdoc_number}_c{section_counter}_{j}_{child_hash}"
                child_chunk = {
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "text": child_text,
                    "heading": section["heading"],
                    "child_index": j,
                    "metadata": {**metadata, "tdoc_number": tdoc_number},
                }
                chunks.append(child_chunk)
        else:
            # Section is small enough to be its own child chunk
            child_hash = hashlib.md5(section["text"][:200].encode()).hexdigest()[:8]
            child_id = f"{tdoc_number}_c{section_counter}_0_{child_hash}"
            child_chunk = {
                "chunk_id": child_id,
                "parent_id": parent_id,
                "chunk_type": "child",
                "text": section["text"],
                "heading": section["heading"],
                "child_index": 0,
                "metadata": {**metadata, "tdoc_number": tdoc_number},
            }
            chunks.append(child_chunk)

        section_counter += 1

    return chunks


def process_all_chunks(
    metadata_path: str, texts_dir: str, output_dir: str
) -> dict:
    """Process all Tdocs and create chunk files.

    Returns stats dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_path) as f:
        records = [json.loads(line) for line in f]

    total_parents = 0
    total_children = 0
    total_tdocs = 0
    all_chunks = []

    for record in records:
        tdoc_num = record["tdoc_number"]
        text_path = os.path.join(texts_dir, f"{tdoc_num}.txt")

        if not os.path.exists(text_path):
            continue

        text = Path(text_path).read_text(encoding="utf-8")
        if not text.strip():
            continue

        # Build metadata for chunks
        meta = {
            "meeting_id": record["meeting_id"],
            "meeting_number": record["meeting_number"],
            "working_group": record["working_group"],
            "companies": record["companies"],
            "title": record["title"],
            "agenda_item": record["agenda_item"],
            "doc_type": record["doc_type"],
        }

        chunks = chunk_document(tdoc_num, text, meta)
        total_tdocs += 1

        # Save per-Tdoc chunk file
        chunk_path = output_path / f"{tdoc_num}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        parents = [c for c in chunks if c["chunk_type"] == "parent"]
        children = [c for c in chunks if c["chunk_type"] == "child"]
        total_parents += len(parents)
        total_children += len(children)
        all_chunks.extend(chunks)

    # Save combined chunks file for indexer
    combined_path = output_path / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    stats = {
        "total_tdocs": total_tdocs,
        "total_chunks": len(all_chunks),
        "total_parents": total_parents,
        "total_children": total_children,
        "avg_chunks_per_tdoc": len(all_chunks) / total_tdocs if total_tdocs else 0,
        "avg_children_per_tdoc": total_children / total_tdocs if total_tdocs else 0,
    }

    logger.info(
        f"Chunking: {total_tdocs} Tdocs, {total_parents} parents, "
        f"{total_children} children ({stats['avg_children_per_tdoc']:.1f} avg/tdoc)"
    )
    return stats
