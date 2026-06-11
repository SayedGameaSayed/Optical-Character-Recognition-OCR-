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

# Fields known to NOT be name/address/id — filter out
NOISE_KEYWORDS: set = {
    "جمهورية", "مصر", "العربية", "بطاقة", "تحقيق", "الشخصية",
    "ذكر", "انثى", "أنثى", "مسلم", "مسيحي", "أعزب", "متزوج",
    "مطلق", "أرمل", "طالب", "موظف", "عامل", "تاجر", "مهندس",
    "طبيب", "سيدة", "السيدة", "الأستاذ", "الدكتور",
    "البطاقة", "سارية", "حتى", "توقيع", "الإصدار", "الرقم",
    "is", "no", "date", "birth", "sex", "religion", "status",
    "occupation", "signature", "expiry",
}

# National ID pattern: exactly 14 digits (after conversion)
NATIONAL_ID_PATTERN = re.compile(r"^\d{14}$")


def _to_western(text: str) -> str:
    """Convert Eastern Arabic numerals to Western digits in-place."""
    return text.translate(EASTERN_TO_WESTERN)


def _is_noise(text: str) -> bool:
    """Check if a text fragment is a known noise field."""
    cleaned = text.strip().lower()
    # Remove diacritics and common Arabic ligature artifacts for matching
    cleaned = re.sub(r"[\u064B-\u065F\u0670]", "", cleaned)
    for kw in NOISE_KEYWORDS:
        if kw in cleaned:
            return True
    return False


def _extract_national_id(candidates: List[str]) -> Optional[str]:
    """Search for exactly 14 consecutive digits in any candidate."""
    for text in candidates:
        western = _to_western(text)
        digits_only = re.sub(r"[^0-9]", "", western)
        if NATIONAL_ID_PATTERN.match(digits_only):
            return digits_only
    return None


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
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        if _is_noise(cleaned):
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
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 5:
            continue
        if cleaned in name:
            continue
        western = _to_western(cleaned)
        digits_only = re.sub(r"[^0-9]", "", western)
        if national_id and digits_only == national_id:
            continue
        if digits_only and len(digits_only) >= 10:
            continue
        if _is_noise(cleaned):
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
