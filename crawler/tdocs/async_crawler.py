"""Async version of Tdoc crawler for faster downloads"""

import asyncio
import aiohttp
import os
import zipfile
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import json
from bs4 import BeautifulSoup
import re


@dataclass
class TdocInfo:
    tdoc_number: str
    filename: str
    url: str
    meeting_id: str
    working_group: str
    local_path: str = None
    extracted_files: List[str] = None


class AsyncTdocCrawler:
    """Async crawler for faster parallel downloads"""

    def __init__(self, max_concurrent: int = 5, data_dir: str = None):
        self.max_concurrent = max_concurrent
        self.data_dir = Path(data_dir or "/Users/dhl/3gpp-agent/data/raw")
        self.semaphore = None

    async def parse_meeting_page(self, session: aiohttp.ClientSession, url: str) -> List[TdocInfo]:
        """Parse meeting page and extract document list"""
        print(f"Fetching: {url}")
        async with session.get(url) as response:
            html = await response.text()

        soup = BeautifulSoup(html, 'lxml')
        documents = []
        meeting_id = self._extract_meeting_id(url)

        for link in soup.find_all('a', class_='file'):
            href = link.get('href', '')
            filename = link.text.strip()

            if not filename or not href:
                continue

            if not filename.lower().endswith('.zip'):
                continue

            tdoc_number = self._extract_tdoc_number(filename)
            if not tdoc_number:
                continue

            working_group = self._extract_working_group(filename)

            documents.append(TdocInfo(
                tdoc_number=tdoc_number,
                filename=filename,
                url=href,
                meeting_id=meeting_id,
                working_group=working_group
            ))

        print(f"Found {len(documents)} documents")
        return documents

    async def download_document(self, session: aiohttp.ClientSession, tdoc: TdocInfo,
                                output_dir: Path) -> bool:
        """Download a single document with rate limiting"""
        async with self.semaphore:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / tdoc.filename

            if output_path.exists():
                tdoc.local_path = str(output_path)
                return True

            try:
                async with session.get(tdoc.url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        tdoc.local_path = str(output_path)
                        print(f"  [DOWN] {tdoc.tdoc_number}")
                        return True
                    else:
                        print(f"  [ERR] {tdoc.tdoc_number}: HTTP {response.status}")
                        return False
            except Exception as e:
                print(f"  [ERR] {tdoc.tdoc_number}: {e}")
                return False

    def extract_document(self, tdoc: TdocInfo) -> List[str]:
        """Extract zip file"""
        if not tdoc.local_path or not os.path.exists(tdoc.local_path):
            return []

        local_path = Path(tdoc.local_path)
        if local_path.suffix.lower() != '.zip':
            tdoc.extracted_files = [str(local_path)]
            return [str(local_path)]

        extract_dir = local_path.parent / local_path.stem
        try:
            with zipfile.ZipFile(local_path, 'r') as zf:
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)

            extracted = [str(f) for f in extract_dir.glob('*') if f.is_file()]
            tdoc.extracted_files = extracted
            return extracted
        except Exception as e:
            print(f"  [ERR] Extract {tdoc.tdoc_number}: {e}")
            return []

    async def crawl_meeting(self, meeting_url: str, limit: int = None) -> List[TdocInfo]:
        """Crawl all documents from a meeting"""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession() as session:
            documents = await self.parse_meeting_page(session, meeting_url)

            if limit:
                documents = documents[:limit]
                print(f"Limiting to {limit} documents")

            output_dir = self.data_dir / documents[0].meeting_id

            print(f"\nDownloading {len(documents)} documents (max {self.max_concurrent} concurrent)...")
            tasks = [
                self.download_document(session, tdoc, output_dir)
                for tdoc in documents
            ]
            await asyncio.gather(*tasks)

        print("\nExtracting files...")
        for tdoc in documents:
            if tdoc.local_path:
                self.extract_document(tdoc)

        return documents

    def _extract_tdoc_number(self, filename: str) -> str:
        match = re.match(r'([A-Z]\d+-\d+)', filename, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_working_group(self, filename: str) -> str:
        match = re.match(r'([A-Z]+\d?)-', filename, re.IGNORECASE)
        return match.group(1).upper() if match else "UNKNOWN"

    def _extract_meeting_id(self, url: str) -> str:
        match = re.search(r'TSGR\d+_\d+|TSG[RSA]\d+', url, re.IGNORECASE)
        return match.group(0).upper() if match else "UNKNOWN"

    def save_manifest(self, documents: List[TdocInfo], output_file: str):
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


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Async 3GPP Tdoc crawler')
    parser.add_argument('--url', default="https://www.3gpp.org/ftp/tsg_ran/WG2_RL2/TSGR2_134/Docs/")
    parser.add_argument('--limit', type=int, help='Limit number of documents')
    parser.add_argument('--concurrent', type=int, default=5, help='Max concurrent downloads')

    args = parser.parse_args()

    crawler = AsyncTdocCrawler(max_concurrent=args.concurrent)
    documents = await crawler.crawl_meeting(args.url, limit=args.limit)

    manifest_path = str(Path(crawler.data_dir) / "manifest.json")
    crawler.save_manifest(documents, manifest_path)

    print(f"\nDone! Downloaded {len(documents)} documents")


if __name__ == '__main__':
    asyncio.run(main())
