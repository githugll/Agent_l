"""Extract and structure metadata from 3GPP Tdoc documents."""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Company name patterns (ordered by specificity - more specific first)
COMPANY_PATTERNS = [
    ("Nokia Bell Labs", ["Nokia Bell Labs"]),
    ("Nokia", ["Nokia", "NOKIA"]),
    ("Ericsson", ["Ericsson", "ERICSSON"]),
    ("Huawei", ["Huawei", "HUAWEI", "HiSilicon"]),
    ("Qualcomm", ["Qualcomm", "QUALCOMM", "QTI"]),
    ("Samsung", ["Samsung", "SAMSUNG"]),
    ("Intel", ["Intel", "INTEL"]),
    ("MediaTek", ["MediaTek", "MTK", "Media Tek"]),
    ("ZTE", ["ZTE"]),
    ("China Mobile", ["China Mobile", "CMCC"]),
    ("NTT Docomo", ["NTT Docomo", "NTT DOCOMO", "DOCOMO", "Docomo"]),
    ("LG Electronics", ["LG Electronics", "LGE"]),
    ("vivo", ["vivo", "VIVO"]),
    ("OPPO", ["OPPO"]),
    ("Xiaomi", ["Xiaomi", "XIAOMI"]),
    ("Futurewei", ["Futurewei"]),
    ("Lenovo", ["Lenovo", "LENOVO"]),
    ("Motorola", ["Motorola", "MOTOROLA"]),
    ("CATT", ["CATT"]),
    ("China Telecom", ["China Telecom"]),
    ("China Unicom", ["China Unicom"]),
    ("Deutsche Telekom", ["Deutsche Telekom", "T-Mobile"]),
    ("InterDigital", ["InterDigital"]),
    ("ITRI", ["ITRI"]),
    ("Spreadtrum", ["Spreadtrum", "UNISOC"]),
    ("NEC", ["NEC"]),
    ("Fujitsu", ["Fujitsu"]),
    ("Sharp", ["Sharp"]),
    ("KDDI", ["KDDI"]),
    ("SoftBank", ["SoftBank", "Softbank"]),
    ("Apple", ["Apple"]),
    ("Google", ["Google"]),
    ("Cisco", ["Cisco"]),
    ("Ofinno", ["Ofinno", "OFINNO"]),
    ("1Finity", ["1Finity", "1FINITY"]),
    ("Sony", ["Sony", "SONY"]),
    ("ASUSTeK", ["ASUSTeK", "ASUS", "Asustek"]),
    ("ETRI", ["ETRI"]),
    ("Panasonic", ["Panasonic", "PANASONIC"]),
    ("CEWiT", ["CEWiT", "CEWIT"]),
    ("Kyocera", ["Kyocera", "KYOCERA"]),
    ("TCL", ["TCL"]),
    ("Jio Platforms", ["Jio Platforms", "Jio"]),
    ("Transsion", ["Transsion"]),
    ("KT Corp", ["KT Corp", "KT Corporation"]),
    ("Tejas Networks", ["Tejas Networks"]),
    ("Fraunhofer", ["Fraunhofer"]),
    ("AT&T", ["AT&T", "ATT"]),
    ("Rakuten", ["Rakuten"]),
    ("CSCN", ["CSCN"]),
    ("Nordic Semiconductor", ["Nordic Semiconductor"]),
    ("Canon", ["Canon"]),
    ("Quectel", ["Quectel", "QUECTEL"]),
    ("DENSO", ["DENSO"]),
    ("Philips", ["Philips"]),
    ("Bosch", ["Robert Bosch", "Bosch"]),
    ("Vodafone", ["Vodafone"]),
    ("SK Telecom", ["SK Telecom"]),
    ("NVIDIA", ["NVIDIA", "Nvidia"]),
    ("Amazon", ["Amazon"]),
    ("Meta", ["Meta Platforms", "Meta"]),
    ("Semtech", ["Semtech"]),
    ("Turkcell", ["Turkcell"]),
    ("Wistron", ["Wistron"]),
    ("Charter Communications", ["Charter Communications"]),
    ("KPN", ["KPN"]),
    ("Pengcheng Laboratory", ["Pengcheng Laboratory", "PengCheng Laboratory", "PCL"]),
    ("National Taiwan University", ["National Taiwan University", "NTU"]),
    ("BUPT", ["Beijing University of Posts"]),
    ("CableLabs", ["CableLabs"]),
    ("Rohde & Schwarz", ["Rohde & Schwarz"]),
    ("AUMOVIO", ["AUMOVIO"]),
    ("ITL", ["ITL"]),
    ("CENC", ["CENC"]),
    ("Eutelsat", ["Eutelsat"]),
    ("Airbus", ["Airbus"]),
    ("Hispasat", ["Hispasat"]),
]

# Build flat lookup
_COMPANY_LOOKUP = [(pat, name) for name, pats in COMPANY_PATTERNS for pat in pats]


def _match_company(text: str, companies: set) -> bool:
    """Try to match company names in text. Returns True if match found."""
    text_lower = text.lower()
    for pattern, name in _COMPANY_LOOKUP:
        if pattern.lower() in text_lower:
            companies.add(name)
            return True
    return False


def _extract_source_company(text: str, tdoc_number: str = "") -> list[str]:
    """Extract company names from the Source field in Tdoc header or filename."""
    companies: set = set()

    # Look for the Source line in the header (first 2000 chars)
    header = text[:2000]

    # Try multiple Source patterns
    patterns = [
        r"Source\s*:\s*(.+?)(?:\n\n|\nTitle|\nAgenda|\nRevision|\nDocument\s|Source\s*:)",
        r"Source\s*:\s*(.+?)(?:\n\n|\n\n\n|  {2,})",
    ]

    for pattern in patterns:
        m = re.search(pattern, header, re.IGNORECASE | re.DOTALL)
        if m:
            source_text = m.group(1).strip()
            source_text = source_text.split("\n")[0].strip()
            # Handle comma-separated multiple sources
            for part in re.split(r",\s*|\s+&\s+|\s+and\s+", source_text):
                part = part.strip()
                if len(part) < 2:
                    continue
                if _match_company(part, companies):
                    break
            if companies:
                break

    return sorted(companies) if companies else ["Unknown"]


def _extract_title(text: str) -> str:
    """Extract title from Tdoc header."""
    title_match = re.search(
        r"Title\s*:\s*(.+?)(?:\n\n|\nAgenda|\nDocument\s)",
        text[:2000], re.IGNORECASE | re.DOTALL
    )
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        return title[:300]
    return ""


def _extract_agenda_item(text: str) -> str:
    """Extract agenda item from Tdoc header."""
    agenda_match = re.search(
        r"Agenda\s*[Ii]tem\s*:\s*(.+?)(?:\n)",
        text[:2000], re.IGNORECASE
    )
    if agenda_match:
        return agenda_match.group(1).strip()
    return ""


def _extract_tdoc_type(text: str) -> str:
    """Determine document type from content."""
    text_lower = text[:2000].lower()
    if "cr cover sheet" in text_lower or "change request" in text_lower:
        return "CR"
    elif "revision of" in text_lower:
        return "Revision"
    elif "ls on" in text_lower or "letter of" in text_lower:
        return "LS"
    elif "draft report" in text_lower:
        return "Draft Report"
    elif "discussion" in text_lower:
        return "Discussion"
    elif "proposal" in text_lower or "way forward" in text_lower:
        return "Proposal"
    else:
        return "Other"


def extract_metadata(tdoc_number: str, text_path: str, manifest_entry: dict) -> dict:
    """Extract structured metadata for a single Tdoc."""
    text = Path(text_path).read_text(encoding="utf-8") if os.path.exists(text_path) else ""

    meeting_id = manifest_entry.get("meeting_id", "")
    meeting_number = 0
    match = re.search(r"(\d+)$", meeting_id)
    if match:
        meeting_number = int(match.group(1))

    wg_match = re.match(r"([A-Z]+\d?)", tdoc_number)
    working_group = wg_match.group(1) if wg_match else ""

    return {
        "tdoc_number": tdoc_number,
        "filename": manifest_entry.get("filename", ""),
        "meeting_id": meeting_id,
        "meeting_number": meeting_number,
        "working_group": working_group,
        "companies": _extract_source_company(text, tdoc_number),
        "title": _extract_title(text),
        "agenda_item": _extract_agenda_item(text),
        "doc_type": _extract_tdoc_type(text) if text else "Unknown",
        "char_count": len(text),
        "word_count": len(text.split()) if text else 0,
        "text_path": str(text_path),
    }


def process_all_metadata(manifest_path: str, texts_dir: str, output_path: str) -> dict:
    """Process all Tdocs and write metadata.jsonl."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    docs = manifest["documents"]
    manifest_lookup = {d["tdoc_number"]: d for d in docs}

    total = 0
    identified = 0
    results = []

    for tdoc_num, entry in manifest_lookup.items():
        text_path = os.path.join(texts_dir, f"{tdoc_num}.txt")
        meta = extract_metadata(tdoc_num, text_path, entry)
        results.append(meta)
        total += 1
        if meta["companies"] != ["Unknown"]:
            identified += 1

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for meta in results:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    stats = {
        "total": total,
        "identified_companies": identified,
        "unknown_companies": total - identified,
        "identification_rate": identified / total * 100 if total > 0 else 0,
    }

    logger.info(
        f"Metadata: {total} total, {identified} identified "
        f"({stats['identification_rate']:.1f}%), "
        f"{total - identified} unknown"
    )
    return stats
