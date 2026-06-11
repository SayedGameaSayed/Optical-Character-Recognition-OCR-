from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCREngine:
    """Thin wrapper around PaddleOCR for Arabic text extraction."""

    def __init__(self, lang: str = "ar") -> None:
        self._ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=6,
        )

    def extract(self, image: np.ndarray) -> List[Tuple[str, float, List[List[int]]]]:
        """Run OCR on a preprocessed image.

        Args:
            image: Binary or grayscale image (H, W) or BGR (H, W, 3).

        Returns:
            List of (text, confidence, bounding_box) tuples sorted roughly
            top-to-bottom by the bounding box centroid.
        """
        result = self._ocr.ocr(image)
        if not result or not result[0]:
            return []

        parsed = []
        for line in result[0]:
            bbox, (text, conf) = line[0], line[1]
            parsed.append((text, conf, bbox))

        # Sort by vertical position (centroid y) then horizontal (centroid x)
        parsed.sort(key=lambda x: (np.mean([p[1] for p in x[2]]),
                                   np.mean([p[0] for p in x[2]])))
        return parsed
