"""3GPP Spec Crawler - Downloads specification ZIPs from 3GPP FTP.

Each ZIP in https://www.3gpp.org/ftp/Specs/2025-12/Rel-19/38_series/
contains exactly one PDF. We filter to the target 8 specs and extract the PDFs.
"""

import hashlib
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import (
    SPEC_BASE_URL,
    TARGET_SPECS,
    RAW_DIR,
    MAX_RETRIES,
    DOWNLOAD_DELAY,
    TIMEOUT,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


@dataclass
class SpecInfo:
    """Represents a 3GPP specification."""
    spec_number: str       # e.g. "38.321"
    version: str           # e.g. "i20"
    filename: str          # e.g. "38321-i20.zip"
    zip_url: str
    zip_path: Optional[str] = None   # local path to downloaded ZIP
    pdf_path: Optional[str] = None   # local path to extracted PDF
    status: str = "pending"          # pending | downloaded | extracted | failed

    @property
    def series(self) -> str:
        return self.spec_number.split(".")[0]

    @property
    def spec_id(self) -> str:
        """Numeric spec id used in filenames: 38321."""
        return self.spec_number.replace(".", "")

    def manifest_entry(self) -> dict:
        return {
            "spec_number": self.spec_number,
            "version": self.version,
            "filename": self.filename,
            "zip_url": self.zip_url,
            "zip_path": self.zip_path,
            "pdf_path": self.pdf_path,
            "status": self.status,
        }


class SpecCrawler:
    """Crawler for 3GPP specifications."""

    # Regex: 38211-i20.zip → spec=38211, version=i20
    ZIP_RE = re.compile(r"^(\d{5})-(gto|gtb|g20|i\d{2}|j\d{2}|[a-z]\d{2})\.zip$", re.IGNORECASE)
    # The version pattern is complex; any 2-char suffix after dash works.
    ZIP_RE_FLEXIBLE = re.compile(r"^(\d{5})-([a-z]\d{2,})\.zip$", re.IGNORECASE)

    def __init__(
        self,
        base_url: str = None,
        raw_dir: str = None,
        target_specs: list = None,
    ):
        self.base_url = base_url or SPEC_BASE_URL
        self.raw_dir = Path(raw_dir or RAW_DIR)
        self.target_specs = set(target_specs or TARGET_SPECS)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    # ── Public API ─────────────────────────────────────────────────────────────

    def crawl_all(self, spec_list: list = None) -> list[SpecInfo]:
        """Full pipeline: discover → filter → download → extract.
        Returns list of SpecInfo for target specs."""
        specs = self.discover_target_specs(spec_list)
        if not specs:
            logger.warning("No target specs found on FTP, check network or spec list")
            return []

        for spec in specs:
            self._download_spec(spec)
            if spec.status == "downloaded":
                self._extract_pdf(spec)

        return specs

    def save_manifest(self, specs: list, output_path: str = None):
        """Save spec manifest JSON."""
        if output_path is None:
            output_path = str(self.raw_dir / "manifest.json")
        manifest = {
            "base_url": self.base_url,
            "target_specs": sorted(self.target_specs),
            "total": len(specs),
            "specs": [s.manifest_entry() for s in specs],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"Manifest saved: {output_path}")

    def load_manifest(self, path: str = None) -> list[SpecInfo]:
        """Load manifest and return SpecInfo list."""
        if path is None:
            path = str(self.raw_dir / "manifest.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        specs = []
        for e in m.get("specs", []):
            s = SpecInfo(
                spec_number=e["spec_number"],
                version=e["version"],
                filename=e["filename"],
                zip_url=e["zip_url"],
                zip_path=e.get("zip_path"),
                pdf_path=e.get("pdf_path"),
                status=e.get("status", "pending"),
            )
            specs.append(s)
        return specs

    # ── Discovery ──────────────────────────────────────────────────────────────

    def discover_all(self) -> list[SpecInfo]:
        """Parse the 38_series HTML page and return all spec entries."""
        url = f"{self.base_url}/38_series/"
        logger.info(f"Fetching: {url}")
        try:
            resp = self.session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        specs = []
        for link in soup.find_all("a", class_="file"):
            href = link.get("href", "")
            filename = link.text.strip()
            if not filename.lower().endswith(".zip"):
                continue
            spec = self._parse_zip_filename(filename, href)
            if spec:
                specs.append(spec)

        logger.info(f"Found {len(specs)} spec ZIPs on FTP")
        return specs

    def discover_target_specs(self, spec_list: list = None) -> list[SpecInfo]:
        """Discover only target specs, picking the latest version for each."""
        targets = set(spec_list or self.target_specs)
        all_specs = self.discover_all()

        # Group by spec number
        by_number: dict[str, list[SpecInfo]] = {}
        for s in all_specs:
            by_number.setdefault(s.spec_number, []).append(s)

        results = []
        for target in sorted(targets):
            candidates = by_number.get(target, [])
            if not candidates:
                logger.warning(f"Target spec {target} not found on FTP")
                continue
            # Pick highest version (alphabetical sort on version string is good enough)
            latest = sorted(candidates, key=lambda x: x.version)[-1]
            logger.info(f"  {target}: picked {latest.version} (from {len(candidates)} versions)")
            results.append(latest)

        return results

    def _parse_zip_filename(self, filename: str, href: str) -> Optional[SpecInfo]:
        """Parse ZIP filename like 38321-i20.zip → spec_number=38.321, version=i20."""
        # Strip path
        filename = Path(filename).name
        match = re.match(r"^(\d{5})-([a-z]\d{2,})\.zip$", filename, re.IGNORECASE)
        if not match:
            return None
        spec_id, version = match.group(1), match.group(2).lower()
        # Derive spec number: 38321 → 38.321
        spec_number = f"{spec_id[:2]}.{spec_id[2:]}"
        zip_url = href if href.startswith("http") else f"{self.base_url}/38_series/{filename}"
        return SpecInfo(
            spec_number=spec_number,
            version=version,
            filename=filename,
            zip_url=zip_url,
        )

    # ── Download ───────────────────────────────────────────────────────────────

    def _download_spec(self, spec: SpecInfo) -> bool:
        """Download a single spec ZIP. Idempotent: skips if already exists."""
        zip_path = self.raw_dir / spec.filename
        if zip_path.exists():
            logger.info(f"  [SKIP] {spec.spec_number} ({spec.version}) already downloaded")
            spec.zip_path = str(zip_path)
            spec.status = "downloaded"
            return True

        logger.info(f"  [DOWN] {spec.spec_number} ({spec.version}) ← {spec.zip_url}")
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(spec.zip_url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                spec.zip_path = str(zip_path)
                spec.status = "downloaded"
                time.sleep(DOWNLOAD_DELAY)
                return True
            except Exception as e:
                logger.warning(f"  [RETRY {attempt+1}] {spec.spec_number}: {e}")
                time.sleep(DOWNLOAD_DELAY * (attempt + 1))

        logger.error(f"  [FAIL] {spec.spec_number} after {MAX_RETRIES} retries")
        spec.status = "failed"
        return False

    # ── Extract ───────────────────────────────────────────────────────────────

    def _extract_pdf(self, spec: SpecInfo) -> bool:
        """Extract the document (PDF or DOCX) from the ZIP into raw_dir."""
        if not spec.zip_path or not Path(spec.zip_path).exists():
            logger.warning(f"No ZIP for {spec.spec_number}")
            spec.status = "failed"
            return False

        zip_path = Path(spec.zip_path)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                # Accept PDF or DOCX
                doc_files = [n for n in names if n.lower().endswith((".pdf", ".docx", ".doc"))]
                if not doc_files:
                    logger.warning(f"No document in {zip_path.name}: {names}")
                    spec.status = "failed"
                    return False

                extract_dir = self.raw_dir / spec.spec_id
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)

                # Find the extracted document
                doc_path = None
                for doc_name in doc_files:
                    candidate = extract_dir / Path(doc_name).name
                    if candidate.exists():
                        doc_path = candidate
                        break
                    nested = extract_dir / doc_name
                    if nested.exists():
                        doc_path = nested
                        break

                if doc_path is None:
                    docs = list(extract_dir.rglob("*"))
                    docs = [d for d in docs if d.suffix.lower() in (".pdf", ".docx", ".doc")]
                    if docs:
                        doc_path = docs[0]

                if doc_path is None:
                    logger.warning(f"Document not found after extraction for {spec.spec_number}")
                    spec.status = "failed"
                    return False

                spec.pdf_path = str(doc_path)
                spec.status = "extracted"
                logger.info(f"  [EXTRACT] {spec.spec_number} → {doc_path.name} ({doc_path.stat().st_size // 1024}KB)")
                return True

        except Exception as e:
            logger.error(f"  [EXTRACT ERR] {spec.spec_number}: {e}")
            spec.status = "failed"
            return False


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Crawl 3GPP specifications")
    parser.add_argument("--spec-list", help="Comma-separated specs, e.g. 38.321,38.331")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    args = parser.parse_args()

    spec_list = None
    if args.spec_list:
        spec_list = [s.strip() for s in args.spec_list.split(",")]

    crawler = SpecCrawler()
    specs = crawler.crawl_all(spec_list=spec_list)
    crawler.save_manifest(specs)

    downloaded = [s for s in specs if s.status == "downloaded"]
    extracted = [s for s in specs if s.status == "extracted"]
    failed = [s for s in specs if s.status == "failed"]
    print(f"\nSummary: {len(specs)} target specs | {len(downloaded)} downloaded | {len(extracted)} extracted | {len(failed)} failed")


if __name__ == "__main__":
    main()
