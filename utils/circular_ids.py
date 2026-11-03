"""SEBI circular numbers, old and new format.

Old: SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/71, SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70
New (2025+): HO/19/34/14(10)2026-AFD-POD3/I/24814/2026, HO/(201)2026-DDHS-POD1/I/24467/2026
PDFs sometimes break the new format with a space ("POD1/ I/146/2025"), so spaces around
slashes are allowed and removed when normalising.
"""
import re

OLD_FORMAT = r"SEBI/[\w/\-]*?(?:CIR|Master)[/\w\-]*\w"
NEW_FORMAT = r"(?<![\w/])HO/[\w()/\-]*?\s?/\s?I\s?/\s?\d+\s?/\s?\d{4}"
CIRCULAR_ID = re.compile(rf"({OLD_FORMAT}|{NEW_FORMAT})")


def normalize_id(raw: str) -> str:
    return re.sub(r"\s+", "", raw)


def find_circular_ids(text: str):
    """Yields (normalized_id, match) for every circular number in the text."""
    for m in CIRCULAR_ID.finditer(text or ""):
        yield normalize_id(m.group(1)), m


def detect_own_id(first_page_text: str) -> str | None:
    """A circular prints its own number at the top of page 1, before 'To,'."""
    header = (first_page_text or "").split("To,")[0]
    for cid, _ in find_circular_ids(header):
        return cid
    return None
