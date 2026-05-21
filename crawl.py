#!/usr/bin/env python3
"""Easy-to-use script for crawling 3GPP Tdoc documents"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawler.tdocs.tdoc_crawler import TdocCrawler


def main():
    print("=" * 60)
    print("3GPP Tdoc Crawler")
    print("=" * 60)

    crawler = TdocCrawler()

    meeting_url = "https://www.3gpp.org/ftp/tsg_ran/WG2_RL2/TSGR2_134/Docs/"

    print(f"\nTarget meeting: RAN2#134")
    print(f"URL: {meeting_url}\n")

    print("Options:")
    print("1. Parse only (no download)")
    print("2. Download first 10 documents")
    print("3. Download all documents")
    print("4. Download custom number")
    print("5. Exit")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        documents = crawler.crawl_meeting(meeting_url, download=False)
    elif choice == "2":
        documents = crawler.crawl_meeting(meeting_url, limit=10)
    elif choice == "3":
        confirm = input("This will download ~800+ documents. Continue? (y/n): ")
        if confirm.lower() == 'y':
            documents = crawler.crawl_meeting(meeting_url)
        else:
            print("Cancelled")
            return
    elif choice == "4":
        count = int(input("Enter number of documents to download: "))
        documents = crawler.crawl_meeting(meeting_url, limit=count)
    else:
        print("Exit")
        return

    manifest_path = os.path.join(crawler.data_dir, "manifest.json")
    crawler.save_manifest(documents, manifest_path)

    print(f"\nDone! Downloaded {len(documents)} documents")
    print(f"Data location: {crawler.data_dir}")


if __name__ == '__main__':
    main()
