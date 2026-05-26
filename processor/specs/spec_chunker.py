"""Hierarchical chunking for 3GPP specifications.

Strategy:
  Parent chunk: one per spec section (supports up to 6-level headings + Annexes A-E)
  Child chunk: ~571 tokens with overlap, linked to parent via parent_id

Spec sections use deeper nesting than TDocs:
  5 → 5.1 → 5.1.1 → 5.1.1.1 → 5.1.1.1.1 → 5.1.1.1.1.1
  Annexes: A.1 → A.1.1 → A.1.1.1
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Section patterns for spec documents
# Main sections: 1, 1.1, 2.3.1, 5.1.2.1, etc. (up to 6 levels)
SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+){0,5})\s+(.{3,120})",
    re.MULTILINE,
)
# Annex sections: A.1, B.2.1, A.1.1.1, etc.
ANNEX_RE = re.compile(
    r"^([A-E]\.\d+(?:\.\d+){0,3})\s+(.{3,120})",
    re.MULTILINE,
)
# Combined regex: matches both main and annex sections
ALL_SECTION_RE = re.compile(
    r"^(?:(\d+(?:\.\d+){0,5})|([A-E]\.\d+(?:\.\d+){0,3}))\s+(.{3,120})",
    re.MULTILINE,
)

CHARS_PER_TOKEN = 3.5
CHILD_MAX_CHARS = int(571 * CHARS_PER_TOKEN)   # ~2000 chars
CHILD_OVERLAP_CHARS = int(64 * CHARS_PER_TOKEN)  # ~224 chars


def _section_depth(section_num: str) -> int:
    """Return nesting depth: "5"→1, "5.1"→2, "5.1.1"→3."""
    return section_num.count(".") + 1


def _section_sort_key(section_num: str):
    """Sort key for section numbers: "5.1" → (5, 1), "A.1" → (99, 0, 1)."""
    if section_num and section_num[0].isalpha():
        # Annex: A.1 → (99, 0, 1), B.2.1 → (99, 1, 2, 1)
        parts = section_num.split(".")
        letter = parts[0][0]
        rest = [int(p) for p in parts[1:]] if len(parts) > 1 else []
        return tuple([99, ord(letter.upper()) - ord("A")] + rest)
    return tuple(int(p) for p in section_num.split("."))


def _split_into_sections(text: str) -> list[dict]:
    """Split spec text into sections based on numbered headings."""
    matches = list(ALL_SECTION_RE.finditer(text))
    if not matches:
        return [{"heading": "Full Document", "section_path": "", "text": text}]

    sections = []

    # Content before first section (cover page, TOC, etc.)
    if matches[0].start() > 100:
        header_text = text[:matches[0].start()].strip()
        if header_text:
            sections.append({
                "heading": "Header",
                "section_path": "header",
                "text": header_text,
            })

    for i, match in enumerate(matches):
        # Extract section number from either group 1 (main) or group 2 (annex)
        section_num = match.group(1) or match.group(2)
        heading = match.group(3).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if section_text:
            sections.append({
                "heading": f"{section_num} {heading}",
                "section_path": section_num,
                "section_title": heading,
                "text": section_text,
            })

    return sections


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if not para.strip():
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_spec(spec_number: str, text: str, spec_meta: dict) -> list[dict]:
    """Create hierarchical chunks for a single spec document.

    Returns list of chunks with:
      chunk_id, parent_id, chunk_type, text, heading,
      metadata (spec_number, section_path, section_title, section_level, ...)
    """
    chunks = []
    sections = _split_into_sections(text)
    section_counter = 0

    for section in sections:
        section_path = section.get("section_path", "")
        section_title = section.get("section_title", section["heading"])
        text_hash = hashlib.md5(section["text"][:200].encode()).hexdigest()[:8]
        parent_id = f"{spec_number}_p{section_counter}_{text_hash}"

        meta = {
            **spec_meta,
            "spec_number": spec_number,
            "section_path": section_path,
            "section_title": section_title[:200],
            "section_level": _section_depth(section_path) if section_path else 0,
        }

        # Parent chunk
        parent_chunk = {
            "chunk_id": parent_id,
            "parent_id": "",
            "chunk_type": "parent",
            "text": section["text"],
            "heading": section["heading"],
            "metadata": meta,
        }
        chunks.append(parent_chunk)

        # Child chunks
        if len(section["text"]) > CHILD_MAX_CHARS:
            child_texts = _chunk_text(section["text"], CHILD_MAX_CHARS, CHILD_OVERLAP_CHARS)
            for j, child_text in enumerate(child_texts):
                child_hash = hashlib.md5(child_text[:200].encode()).hexdigest()[:8]
                child_id = f"{spec_number}_c{section_counter}_{j}_{child_hash}"
                chunks.append({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "text": child_text,
                    "heading": section["heading"],
                    "child_index": j,
                    "metadata": {**meta},
                })
        else:
            child_hash = hashlib.md5(section["text"][:200].encode()).hexdigest()[:8]
            child_id = f"{spec_number}_c{section_counter}_0_{child_hash}"
            chunks.append({
                "chunk_id": child_id,
                "parent_id": parent_id,
                "chunk_type": "child",
                "text": section["text"],
                "heading": section["heading"],
                "child_index": 0,
                "metadata": {**meta},
            })

        section_counter += 1

    return chunks


def process_all_specs(texts_dir: str, output_dir: str, meta_dir: str = None) -> dict:
    """Process all parsed spec texts and create chunks.

    Args:
        texts_dir: Directory with spec .txt files (e.g. data/specs/processed/texts/)
        output_dir: Directory for chunk JSON output
        meta_dir: Directory with spec _meta.json files (defaults to texts_dir)
    """
    texts_path = Path(texts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if meta_dir is None:
        meta_dir = texts_dir
    meta_path = Path(meta_dir)

    all_chunks = []
    total_sections = 0
    total_children = 0
    specs_processed = 0

    # Find all spec .txt files
    txt_files = sorted(texts_path.glob("*.txt"))
    # Exclude _meta.txt files
    txt_files = [f for f in txt_files if not f.name.endswith("_meta.txt")]

    for txt_file in txt_files:
        spec_id = txt_file.stem  # e.g. "38321"
        if not spec_id.startswith("38"):
            continue

        # Derive spec number: 38321 → 38.321
        spec_number = f"{spec_id[:2]}.{spec_id[2:]}"

        text = txt_file.read_text(encoding="utf-8")
        if not text.strip():
            continue

        # Load spec metadata if available
        spec_meta = {}
        meta_file = meta_path / f"{spec_id}_meta.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                spec_meta = json.load(f)

        # Chunk
        chunks = chunk_spec(spec_number, text, spec_meta)

        # Save per-spec chunk file
        chunk_path = output_path / f"{spec_id}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        parents = [c for c in chunks if c["chunk_type"] == "parent"]
        children = [c for c in chunks if c["chunk_type"] == "child"]
        total_sections += len(parents)
        total_children += len(children)
        all_chunks.extend(chunks)
        specs_processed += 1

        logger.info(f"  {spec_number}: {len(parents)} sections, {len(children)} child chunks")

    # Save combined chunks for BM25
    combined_path = output_path / "all_spec_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    stats = {
        "specs_processed": specs_processed,
        "total_sections": total_sections,
        "total_children": total_children,
        "total_chunks": len(all_chunks),
    }
    logger.info(
        f"Chunking: {specs_processed} specs, {total_sections} sections, "
        f"{total_children} child chunks ({len(all_chunks)} total)"
    )
    return stats
