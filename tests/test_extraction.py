from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from app.services.validator import validate_and_parse


def _make_ocr_result(
    texts: List[str],
    conf: float = 0.85,
) -> List[Tuple[str, float, List[List[int]]]]:
    bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]
    return [(t, conf, bbox) for t in texts]


class TestValidator:
    def test_national_id_western_digits(self):
        results = _make_ocr_result([
            "بطاقة تحقيق الشخصية",
            "جمهورية مصر العربية",
            "سيد جامع سید حسین",
            "12345678901234",
            "ابو عموری مركز نجع حمادى - قنا",
        ])
        parsed = validate_and_parse(results)
        assert parsed["national_id"] == "12345678901234"

    def test_national_id_eastern_digits(self):
        results = _make_ocr_result([
            "بطاقة تحقيق الشخصية",
            "۳۰۴۰۹۲۱۲۷۰۱۹۵۵",
        ])
        parsed = validate_and_parse(results)
        assert parsed["national_id"] == "30409212701955"

    def test_national_id_mixed_digits(self):
        results = _make_ocr_result([
            "الرقم القومي ٣٠٤٠٩٢١٢٧٠١٩٥٥",
        ])
        parsed = validate_and_parse(results)
        assert parsed["national_id"] == "30409212701955"

    def test_filter_noise_fields(self):
        results = _make_ocr_result([
            "سيد جامع سید حسین",
            "ذكر",
            "مسلم",
            "أعزب",
            "طالب",
            "٣٠٤٠٩٢١٢٧٠١٩٥٥",
            "ابو عموری مركز نجع حمادى - قنا",
        ])
        parsed = validate_and_parse(results)
        assert "ذكر" not in parsed["name"]
        assert "مسلم" not in parsed["name"]
        assert parsed["national_id"] == "30409212701955"
        assert len(parsed["name"]) > 0

    def test_empty_ocr_results(self):
        parsed = validate_and_parse([])
        assert parsed["name"] == ""
        assert parsed["address"] == ""
        assert parsed["national_id"] == ""
        assert parsed["confidence_score"] == 0.0

    def test_name_extraction(self):
        results = _make_ocr_result([
            "بطاقة تحقيق الشخصية",
            "جمهورية مصر العربية",
            "سيد جامع سید حسین",
            "٣٠٤٠٩٢١٢٧٠١٩٥٥",
            "ابو عموری",
            "مركز نجع حمادى - قنا",
        ])
        parsed = validate_and_parse(results)
        assert "سيد" in parsed["name"]
        assert "حسین" in parsed["name"] or "حسين" in parsed["name"]

    def test_address_extraction(self):
        results = _make_ocr_result([
            "بطاقة تحقيق الشخصية",
            "سيد جامع سید حسین",
            "٣٠٤٠٩٢١٢٧٠١٩٥٥",
            "ابو عموری مركز نجع حمادى - قنا",
        ])
        parsed = validate_and_parse(results)
        assert "نجع" in parsed["address"] or "نجع حمادى" in parsed["address"]

    def test_confidence_score(self):
        results = _make_ocr_result(
            ["سيد جامع", "٣٠٤٠٩٢١٢٧٠١٩٥٥"],
            conf=0.75,
        )
        parsed = validate_and_parse(results)
        assert parsed["confidence_score"] == 0.75

    def test_invalid_national_id_too_short(self):
        results = _make_ocr_result([
            "سيد جامع",
            "1234567890",  # only 10 digits
            "ابو عموری",
        ])
        parsed = validate_and_parse(results)
        assert parsed["national_id"] == ""

    def test_invalid_national_id_too_long(self):
        results = _make_ocr_result([
            "سيد جامع",
            "123456789012345",  # 15 digits
        ])
        parsed = validate_and_parse(results)
        assert parsed["national_id"] == ""

    def test_noise_keyword_filtering(self):
        results = _make_ocr_result([
            "البطاقة سارية حتى ٢٠٢٨/٠١/٠٨",
            "ذكر",
            "مسلم",
            "أعزب",
        ])
        parsed = validate_and_parse(results)
        assert parsed["name"] == ""  # all are noise

    def test_no_arabic_script_ignored(self):
        results = _make_ocr_result([
            "IS1866629",
            "$7..4/.9/1$",
            "سيد جامع",
        ])
        parsed = validate_and_parse(results)
        assert parsed["name"] == "سيد جامع"
