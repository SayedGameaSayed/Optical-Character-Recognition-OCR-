from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Eastern Arabic numerals (two Unicode ranges)
# U+0660-U+0669: Arabic-Indic (used in Arabic)
# U+06F0-U+06F9: Extended Arabic-Indic (used in Persian/Urdu)
_EASTERN_DIGITS = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_WESTERN_DIGITS = "0123456789" * 2
EASTERN_TO_WESTERN = str.maketrans(_EASTERN_DIGITS, _WESTERN_DIGITS)

# Reversed-keyword fragments that indicate PIL/BIDI rendering artifact.
# If any of these appear in a candidate, the entire text is reversed.
_REVERSED_KEYWORDS: set = {
    "ةقاطب",  # بطاقة reversed
    "مسا",    # اسم reversed
    "ونعلت",  # العنوان reversed (partial)
    "سنجلت",  # الجنس reversed (partial)
    "ةيصخشل", # الشخصية reversed (partial)
    "ةنهمللل",# المهنة reversed (partial)
    "ةيءاس",  # سارية reversed
    "ةقاطبلت",# البطاقة reversed (partial)
    "ةيعامت",  # الاجتماعية reversed (partial)
    "ةلااحل", # الحالة reversed (partial)
    "ةنايدلت",# الديانة reversed (partial)
    "ءزعء",   # أعزب reversed (partial)
    "بلاط",   # طالب reversed (partial)
    "ءاءومج", # جمهورية reversed (partial)
    "ركآ",    # أكبر / ذكر reversed (partial)
    "يتح",    # حتى reversed
}

# Stop-words: boilerplate ID-card text and reversed variants.
# Removed at word-level so that real name/address words survive.
STOP_WORDS: set = {
    "بطاقة", "تحقيق", "الشخصية", "جمهورية", "مصر", "العربية",
    "ذكر", "انثى", "مسلم", "مسيحي", "اعزب", "أعزب", "متزوج",
    "طالب", "سارية", "حتى", "الرقم", "القومي",
    # reversed variants
    "ةقاطب", "قيقحت", "ةيصخشلا", "ةيروهمج", "رصم", "ةيبرعلا",
    "ركذ", "ىثنا", "ملسم", "يحيسم", "بزعأ", "جوزتم", "بلاط",
    "ةيراس", "ىتح",
}

def _to_western(text: str) -> str:
    """Convert Eastern Arabic numerals to Western digits in-place."""
    return text.translate(EASTERN_TO_WESTERN)


def _fix_reversed(text: str) -> str:
    """Detect reversed synthetic Arabic text and reverse it.

    PIL/BIDI rendering can produce reversed fragments; if a known
    reversed keyword is found, the whole string is flipped back.
    """
    for kw in _REVERSED_KEYWORDS:
        if kw in text:
            return text[::-1]
    return text


def _remove_stop_words(text: str) -> str:
    """Remove any STOP_WORDS from the text at word level."""
    cleaned = text.strip()
    cleaned = re.sub(r"[\u064B-\u065F\u0670]", "", cleaned)
    words = cleaned.split()
    kept = [w for w in words if w not in STOP_WORDS]
    return " ".join(kept)


def _contains_date(text: str) -> bool:
    """Check if text contains a date pattern like ٢٠٢٨/٠١/٠٨."""
    western = _to_western(text)
    if re.search(r"\d{4}/\d{1,2}/\d{1,2}", western):
        return True
    if re.search(r"/\d", western) or re.search(r"\d/", western):
        return True
    return False


def _is_predominantly_numbers(text: str) -> bool:
    """Check if more than half the characters are digits."""
    western = _to_western(text)
    digits = re.sub(r"\D", "", western)
    if not digits:
        return False
    return len(digits) > len(text) / 2


def _extract_national_id(candidates: List[str]) -> Optional[str]:
    """Aggressively extract a 14-digit national ID from all OCR text.

    PaddleOCR may split the 14-digit number across multiple regions
    or inject spaces/punctuation.  Join all digit runs from every
    candidate, then search for a contiguous 14-digit block.
    """
    raw = " ".join(candidates)
    western = _to_western(raw)
    all_digits = "".join(re.findall(r"\d+", western))
    match = re.search(r"\d{14}", all_digits)
    return match.group(0) if match else None


def _extract_arabic_name(
    candidates: List[str],
    national_id: Optional[str],
) -> str:
    """Reconstruct the Arabic full name by filtering noise and short fragments.

    Heuristic:
      - Take the longest run of consecutive Arabic-only lines that are NOT
        noise keywords, NOT the national ID, and have at least 3 characters.
      - Fall back to the longest single candidate.
    """
    filtered: List[str] = []
    valid_lines: List[str] = []

    for text in candidates:
        cleaned = _fix_reversed(text.strip())
        # 1. Remove stop-words at word level
        cleaned = _remove_stop_words(cleaned)
        if not cleaned or len(cleaned) < 3:
            continue
        # 2. Discard lines containing dates
        if _contains_date(cleaned):
            continue
        # 3. Discard lines that are mostly numbers
        if _is_predominantly_numbers(cleaned):
            continue
        western = _to_western(cleaned)
        digits_only = re.sub(r"[^0-9]", "", western)
        if national_id and digits_only == national_id:
            continue
        # Skip lines that are purely numeric (likely misread national ID)
        if digits_only and len(digits_only) >= 10:
            continue
        # Check that it contains at least some Arabic script
        if not re.search(r"[\u0600-\u06FF]", cleaned):
            continue
        valid_lines.append(cleaned)

    # Try to find address-like patterns (long lines, often contain certain words)
    address_indicators = {"شارع", "ش", "ميدان", "م", "قسم", "مركز", "نجع",
                          "مدينة", "مدينه", "محافظة", "محافظه", "قرية", "قريه",
                          "عزبة", "عزبه", "كفر", "أول", "ثان", "دائرة", "دائره",
                          "ابو", "أبو", "باب"}

    name_lines: List[str] = []
    address_lines: List[str] = []

    for line in valid_lines:
        if any(indicator in line for indicator in address_indicators):
            address_lines.append(line)
        else:
            name_lines.append(line)

    # The name is typically the first few Arabic lines before the address
    # Join up to 4 short-to-medium lines as the name
    name = " ".join(name_lines[:4]).strip()
    if len(name) < 5 and valid_lines:
        name = valid_lines[0]

    return name


def _extract_address(
    candidates: List[str],
    national_id: Optional[str],
    name: str,
) -> str:
    """Extract the address from OCR candidates.

    Typically the address appears after the name, contains location keywords,
    and is a longer continuous string.
    """
    address_indicators = {"شارع", "ش", "ميدان", "م", "قسم", "مركز", "نجع",
                          "مدينة", "مدينه", "محافظة", "محافظه", "قرية", "قريه",
                          "عزبة", "عزبه", "كفر", "أول", "ثان", "دائرة", "دائره",
                          "ابو", "أبو", "باب", "طريق", "ترعة", "ترعه"}

    address_candidates: List[str] = []

    for text in candidates:
        cleaned = _fix_reversed(text.strip())
        cleaned = _remove_stop_words(cleaned)
        if not cleaned or len(cleaned) < 5:
            continue
        if cleaned in name:
            continue
        if _contains_date(cleaned):
            continue
        if _is_predominantly_numbers(cleaned):
            continue
        western = _to_western(cleaned)
        digits_only = re.sub(r"[^0-9]", "", western)
        if national_id and digits_only == national_id:
            continue
        if digits_only and len(digits_only) >= 10:
            continue
        if not re.search(r"[\u0600-\u06FF]", cleaned):
            continue
        address_candidates.append(cleaned)

    # Prefer lines with address indicators
    prioritized = [l for l in address_candidates
                   if any(ind in l for ind in address_indicators)]
    if not prioritized:
        prioritized = address_candidates

    return " | ".join(prioritized[:3]) if prioritized else ""


def validate_and_parse(
    ocr_results: List[Tuple[str, float, List[List[int]]]],
) -> Dict[str, object]:
    """Post-process OCR output into structured fields.

    Args:
        ocr_results: List of (text, confidence, bbox) from OCREngine.

    Returns:
        dict with keys: name, address, national_id, confidence_score.
    """
    if not ocr_results:
        return {
            "name": "",
            "address": "",
            "national_id": "",
            "confidence_score": 0.0,
        }

    candidates = [text for text, conf, _ in ocr_results]

    national_id = _extract_national_id(candidates)
    name = _extract_arabic_name(candidates, national_id)
    address = _extract_address(candidates, national_id, name)

    # Overall confidence: average of all non-noise detections
    confidences = [
        conf for text, conf, _ in ocr_results
        if len(text.strip()) >= 2
    ]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "name": name,
        "address": address,
        "national_id": national_id or "",
        "confidence_score": round(avg_conf, 4),
    }
