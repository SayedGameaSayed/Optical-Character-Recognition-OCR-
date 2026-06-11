"""
End-to-end test: generates a synthetic Egyptian ID card image,
runs the full pipeline (preprocess -> OCR -> validate), prints results.

Usage:
    python test_pipeline.py
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.preprocessor import preprocess_image
from app.services.ocr_engine import OCREngine
from app.services.validator import validate_and_parse

logging.basicConfig(level=logging.INFO)

W, H = 1000, 630
FONT_PATH = "C:/Windows/Fonts/arabtype.ttf"


def prepare_arabic_text(text: str) -> str:
    """Reshape and apply BIDI so Arabic renders correctly connected and RTL."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# Strings that should NOT be reshaped (Latin, digits-only, mixed)
_RAW_STRINGS = {
    "IS1866629", "٣٠٤٠٩٢١٢٧٠١٩٥٥", "٢٠٢٨/٠١/٠٨",
}


def _render_text(text: str) -> str:
    return text if text in _RAW_STRINGS else prepare_arabic_text(text)


def generate_fake_id() -> np.ndarray:
    """Generate a synthetic Egyptian ID card with Arabic text rendered via PIL."""
    img = Image.new("RGB", (W, H), (235, 235, 235))
    draw = ImageDraw.Draw(img)

    font_small = ImageFont.truetype(FONT_PATH, 22)
    font_medium = ImageFont.truetype(FONT_PATH, 28)
    font_large = ImageFont.truetype(FONT_PATH, 36)
    font_id = ImageFont.truetype(FONT_PATH, 44)

    def _draw(text, x, y, font=font_medium, fill=(0, 0, 0)):
        draw.text((x, y), _render_text(text), font=font, fill=fill)

    draw.rectangle([(0, 0), (W, 55)], fill=(45, 45, 130))
    _draw("جمهورية مصر العربية", 480, 12, font_medium, (255, 255, 255))
    _draw("بطاقة تحقيق الشخصية", 760, 14, font_small, (255, 255, 255))

    _draw("الرقم القومي", 50, 90, font_small, (80, 80, 80))
    _draw("٣٠٤٠٩٢١٢٧٠١٩٥٥", 50, 125, font_id, (0, 0, 0))

    _draw("الاسم", 50, 200, font_small, (80, 80, 80))
    _draw("سيد جامع سید حسین", 50, 235, font_large, (0, 0, 0))

    _draw("العنوان", 50, 310, font_small, (80, 80, 80))
    _draw("ابو عموری مركز نجع حمادى - قنا", 50, 345, font_medium, (0, 0, 0))

    noise = [
        ("الديانة", "مسلم"),
        ("الحالة الاجتماعية", "أعزب"),
        ("المهنة", "طالب"),
        ("الجنس", "ذكر"),
    ]
    for i, (label, val) in enumerate(noise):
        y = 420 + i * 42
        _draw(label, 50, y, font_small, (100, 100, 100))
        _draw(val, 220, y, font_small, (100, 100, 100))

    _draw("IS1866629", 780, 500, font_small, (150, 150, 150))
    _draw("البطاقة سارية حتى ٢٠٢٨/٠١/٠٨", 50, 580, font_small, (100, 100, 100))

    np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return np_img


def main():
    print("=" * 60)
    print("Egyptian ID OCR - Pipeline Test")
    print("=" * 60)

    print("\n[1/4] Generating synthetic ID card image...")
    fake_id = generate_fake_id()
    cv2.imwrite("test_fake_id.jpg", fake_id)
    print("      Saved: test_fake_id.jpg")

    print("[2/4] Running pre-processing pipeline...")
    processed = preprocess_image(fake_id, debug=True, debug_dir="debug_output")
    print("      Output shape:", processed.shape)

    print("[3/4] Running OCR engine...")
    ocr = OCREngine()
    results = ocr.extract(processed)
    print(f"      Found {len(results)} text regions")
    for text, conf, bbox in results:
        print(f"      [{conf:.3f}] {text}")

    print("[4/4] Running validation...")
    parsed = validate_and_parse(results)
    print()
    print("=" * 60)
    print("EXTRACTION RESULT")
    print("=" * 60)
    print(f"  Name:          {parsed['name']}")
    print(f"  Address:       {parsed['address']}")
    print(f"  National ID:   {parsed['national_id']}")
    print(f"  Confidence:    {parsed['confidence_score']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
