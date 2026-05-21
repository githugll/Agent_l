"""Spec pipeline configuration."""

import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, "data")

# ── Paths ──────────────────────────────────────────────────
RAW_DIR = os.path.join(DATA_DIR, "specs", "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "specs", "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "specs", "chroma_db")

# ── Crawl ──────────────────────────────────────────────────
SPEC_BASE_URL = "https://www.3gpp.org/ftp/Specs/2025-12/Rel-19"

TARGET_SPECS = [
    "38.211",  # NR; Physical channels and modulation
    "38.212",  # NR; Multiplexing and channel coding
    "38.213",  # NR; Physical layer procedures for control
    "38.214",  # NR; Physical layer procedures for data
    "38.300",  # NR; Overall description
    "38.321",  # NR; MAC layer protocol
    "38.322",  # NR; RLC layer protocol
    "38.331",  # NR; RRC protocol
]

DOWNLOAD_DELAY = 0.5
MAX_RETRIES = 3
TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
