#!/usr/bin/env python3
"""3GPP Tdoc Crawler - Downloads meeting documents from 3GPP FTP server"""

import os
import re
import time
import zipfile
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from typing import List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class TdocInfo:
    """Represents a 3GPP Tdoc (Technical Document)"""
    tdoc_number: str
    filename: str
    url: str
    meeting_id: str
    working_group: str
    local_path: Optional[str] = None
    extracted_files: List[str] = None

    def __post_init__(self):
        if self.extracted_files is None:
            self.extracted_files = []


class TdocCrawler:
    """Crawler for 3GPP meeting documents"""

    def __init__(self, base_url: str = None, data_dir: str = None):
        from .config import BASE_URL, RAW_DIR
        self.base_url = base_url or BASE_URL
        self.data_dir = Path(data_dir or RAW_DIR)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def parse_meeting_page(self, meeting_url: str) -> List[TdocInfo]:
        """Parse meeting page and extract document list"""
        print(f"Fetching: {meeting_url}")
        response = self.session.get(meeting_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')

        documents = []
        meeting_id = self._extract_meeting_id(meeting_url)

        for link in soup.find_all('a', class_='file'):
            href = link.get('href', '')
            filename = link.text.strip()

            if not filename or not href:
                continue

            if not self._is_tdoc_file(filename):
                continue

            tdoc_number = self._extract_tdoc_number(filename)
            if not tdoc_number:
                continue

            working_group = self._extract_working_group(filename)

            doc = TdocInfo(
                tdoc_number=tdoc_number,
                filename=filename,
                url=href,
                meeting_id=meeting_id,
                working_group=working_group
            )
            documents.append(doc)

        print(f"Found {len(documents)} documents")
        return documents

    def _is_tdoc_file(self, filename: str) -> bool:
        """Check if file is a valid Tdoc"""
        valid_extensions = ('.zip', '.doc', '.docx', '.pdf')
        return filename.lower().endswith(valid_extensions)

    def _extract_tdoc_number(self, filename: str) -> Optional[str]:
        """Extract Tdoc number from filename (e.g., R2-2602904)"""
        match = re.match(r'([A-Z]\d+-\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def _extract_working_group(self, filename: str) -> str:
        """Extract working group from filename"""
        match = re.match(r'([A-Z]+\d?)-', filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "UNKNOWN"

    def _extract_meeting_id(self, url: str) -> str:
        """Extract meeting ID from URL"""
        match = re.search(r'TSGR\d+_\d+|TSG[RSA]\d+', url, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return "UNKNOWN"

    def download_document(self, tdoc: TdocInfo, output_dir: Path = None) -> Optional[Path]:
        """Download a single document"""
        output_dir = output_dir or self.data_dir / tdoc.meeting_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / tdoc.filename

        if output_path.exists():
            print(f"  [SKIP] {tdoc.tdoc_number} already exists")
            tdoc.local_path = str(output_path)
            return output_path

        print(f"  [DOWN] {tdoc.tdoc_number}")
        try:
            response = self.session.get(tdoc.url, timeout=60, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            tdoc.local_path = str(output_path)
            time.sleep(0.5)
            return output_path

        except Exception as e:
            print(f"  [ERR] Failed to download {tdoc.tdoc_number}: {e}")
            return None

    def extract_document(self, tdoc: TdocInfo) -> List[Path]:
        """Extract zip file and return list of extracted files"""
        if not tdoc.local_path:
            return []

        local_path = Path(tdoc.local_path)
        if not local_path.exists():
            return []

        extract_dir = local_path.parent / local_path.stem

        if not local_path.suffix.lower() == '.zip':
            tdoc.extracted_files = [str(local_path)]
            return [local_path]

        print(f"  [EXTRACT] {tdoc.tdoc_number}")
        try:
            with zipfile.ZipFile(local_path, 'r') as zf:
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)

            extracted = list(extract_dir.glob('*'))
            tdoc.extracted_files = [str(f) for f in extracted if f.is_file()]
            return extracted

        except Exception as e:
            print(f"  [ERR] Failed to extract {tdoc.tdoc_number}: {e}")
            return []

    def crawl_meeting(self, meeting_url: str, download: bool = True, extract: bool = True,
                      limit: int = None) -> List[TdocInfo]:
        """Crawl all documents from a meeting"""
        documents = self.parse_meeting_page(meeting_url)

        if limit:
            documents = documents[:limit]
            print(f"Limiting to {limit} documents")

        if download:
            print(f"\nDownloading {len(documents)} documents...")
            for i, tdoc in enumerate(documents, 1):
                print(f"[{i}/{len(documents)}]", end=" ")
                self.download_document(tdoc)

                if extract and tdoc.local_path:
                    self.extract_document(tdoc)

        return documents

    def save_manifest(self, documents: List[TdocInfo], output_file: str):
        """Save document list to JSON manifest"""
        manifest = {
            'total': len(documents),
            'documents': [
                {
                    'tdoc_number': d.tdoc_number,
                    'filename': d.filename,
                    'url': d.url,
                    'meeting_id': d.meeting_id,
                    'working_group': d.working_group,
                    'local_path': d.local_path,
                    'extracted_files': d.extracted_files
                }
                for d in documents
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest saved to: {output_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Crawl 3GPP Tdoc documents')
    parser.add_argument('url', nargs='?', help='Meeting URL to crawl')
    parser.add_argument('--meeting', '-m', help='Meeting ID from config')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of documents')
    parser.add_argument('--no-download', action='store_true', help='Only parse, do not download')
    parser.add_argument('--no-extract', action='store_true', help='Do not extract zip files')

    args = parser.parse_args()

    crawler = TdocCrawler()

    if args.meeting:
        from .config import MEETINGS
        if args.meeting not in MEETINGS:
            print(f"Unknown meeting: {args.meeting}")
            print(f"Available: {list(MEETINGS.keys())}")
            return

        meeting = MEETINGS[args.meeting]
        url = crawler.base_url + meeting['path']
    elif args.url:
        url = args.url
    else:
        url = "https://www.3gpp.org/ftp/tsg_ran/WG2_RL2/TSGR2_134/Docs/"

    documents = crawler.crawl_meeting(
        url,
        download=not args.no_download,
        extract=not args.no_extract,
        limit=args.limit
    )

    manifest_path = Path(crawler.data_dir) / "manifest.json"
    crawler.save_manifest(documents, str(manifest_path))


if __name__ == '__main__':
    main()
