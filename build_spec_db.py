#!/usr/bin/env python3
"""Build the Spec vector database: crawl → parse → chunk → index.

Usage:
    python build_spec_db.py                        # Full pipeline
    python build_spec_db.py --skip-crawl            # Skip download, re-process existing
    python build_spec_db.py --spec-list 38.321,38.331  # Process specific specs
"""

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build 3GPP Spec vector database")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip download step")
    parser.add_argument("--spec-list", help="Comma-separated specs, e.g. 38.321,38.331")
    args = parser.parse_args()

    spec_list = None
    if args.spec_list:
        spec_list = [s.strip() for s in args.spec_list.split(",")]

    # Ensure running from project root
    if not os.path.exists("data"):
        print("Error: run from project root (where data/ exists)")
        sys.exit(1)

    # ── Step 1: Crawl specs ──────────────────────────────────────────────────
    if not args.skip_crawl:
        logger.info("=" * 60)
        logger.info("Step 1: Crawling specs from 3GPP FTP...")
        logger.info("=" * 60)
        from crawler.specs.spec_crawler import SpecCrawler
        crawler = SpecCrawler()
        specs = crawler.crawl_all(spec_list=spec_list)
        crawler.save_manifest(specs)
        extracted = [s for s in specs if s.status == "extracted"]
        failed = [s for s in specs if s.status == "failed"]
        logger.info(f"Crawl summary: {len(extracted)} extracted, {len(failed)} failed")
        if not extracted:
            logger.error("No specs extracted, aborting")
            sys.exit(1)
    else:
        logger.info("Skipping crawl (--skip-crawl)")
        from crawler.specs.spec_crawler import SpecCrawler
        crawler = SpecCrawler()
        specs = crawler.load_manifest()
        logger.info(f"Loaded {len(specs)} specs from manifest")

    # ── Step 2: Parse PDFs ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Parsing spec PDFs...")
    logger.info("=" * 60)
    from processor.specs.spec_parser import SpecParser
    from crawler.specs.config import RAW_DIR, PROCESSED_DIR

    sp = SpecParser(RAW_DIR, PROCESSED_DIR)
    parse_stats = sp.process_all(os.path.join(RAW_DIR, "manifest.json"))
    logger.info(f"Parse summary: {parse_stats['success']}/{parse_stats['total']} specs, "
                f"{parse_stats['total_chars']:,} total chars")

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Chunking spec text...")
    logger.info("=" * 60)
    from processor.specs.spec_chunker import process_all_specs

    texts_dir = os.path.join(PROCESSED_DIR, "texts")
    chunks_dir = os.path.join(PROCESSED_DIR, "chunks")
    chunk_stats = process_all_specs(texts_dir, chunks_dir)
    logger.info(f"Chunk summary: {chunk_stats['specs_processed']} specs, "
                f"{chunk_stats['total_sections']} sections, "
                f"{chunk_stats['total_children']} child chunks")

    # ── Step 4: Build vectorstore ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4: Building spec ChromaDB vectorstore...")
    logger.info("=" * 60)
    from processor.specs.spec_indexer import build_spec_vectorstore, verify_spec_vectorstore
    from crawler.specs.config import CHROMA_DIR

    build_spec_vectorstore(
        chunks_file=os.path.join(chunks_dir, "all_spec_chunks.json"),
        persist_dir=CHROMA_DIR,
    )
    verify_spec_vectorstore(CHROMA_DIR)

    logger.info("=" * 60)
    logger.info("Spec database build complete!")
    logger.info(f"  Vector DB: {CHROMA_DIR}")
    logger.info(f"  Specs: {chunk_stats['specs_processed']}")
    logger.info(f"  Total chunks: {chunk_stats['total_chunks']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
