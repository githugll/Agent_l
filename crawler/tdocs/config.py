"""TDoc pipeline configuration."""

import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, "data")

# ── Paths ──────────────────────────────────────────────────
RAW_DIR = os.path.join(DATA_DIR, "tdocs", "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "tdocs", "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "tdocs", "chroma_db")

# ── Crawl ──────────────────────────────────────────────────
BASE_URL = "https://www.3gpp.org/ftp"

MEETINGS = {
    "RAN2_134": {
        "path": "/tsg_ran/WG2_RL2/TSGR2_134/Docs",
        "working_group": "RAN2",
        "meeting_number": 134,
    },
}

DOWNLOAD_DELAY = 0.5
MAX_RETRIES = 3
TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
