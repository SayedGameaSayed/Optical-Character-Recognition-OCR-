from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1000
TARGET_HEIGHT = 630
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
ADAPTIVE_BLOCK_SIZE = 21
ADAPTIVE_C = 4
DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21
MORPH_KERNEL_SIZE = (3, 3)
DEFAULT_DEBUG_DIR = "debug_output"


def load_image(source: bytes | np.ndarray | str) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, bytes):
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        return img
    if isinstance(source, str | Path):
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from path: {source}")
        return img
    raise TypeError(f"Unsupported source type: {type(source)}")


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _detect_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        logger.warning("No contours found in image")
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    total_area = h * w
    MIN_CONTOUR_AREA = 0.6 * total_area

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2).astype(np.float32))

    logger.warning("No quadrilateral contour found; skipping perspective warp")
    return None


def perspective_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    dst_pts = np.array(
        [
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(
        image, M, (TARGET_WIDTH, TARGET_HEIGHT),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return warped


def _denoise_and_binarize(card: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card

    denoised = cv2.fastNlMeansDenoising(
        gray, h=DENOISE_H,
        templateWindowSize=DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=DENOISE_SEARCH_WINDOW,
    )

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


def _save_debug(
    image: np.ndarray, name: str, debug_dir: str = DEFAULT_DEBUG_DIR
) -> str:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / name
    cv2.imwrite(str(filepath), image)
    logger.debug("Saved debug image: %s", filepath)
    return str(filepath)


def _enhance_for_ocr(card: np.ndarray) -> np.ndarray:
    """Gentle denoising only — PaddleOCR handles its own internal preprocessing."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card

    denoised = cv2.fastNlMeansDenoising(
        gray, h=DENOISE_H,
        templateWindowSize=DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=DENOISE_SEARCH_WINDOW,
    )

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def preprocess_image(
    image: np.ndarray,
    debug: bool = False,
    debug_dir: str = DEFAULT_DEBUG_DIR,
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("Input image is empty")

    if debug:
        _save_debug(image, "01_original.jpg", debug_dir)

    contour = _detect_card_contour(image)
    if contour is not None:
        warped = perspective_warp(image, contour)
        if debug:
            _save_debug(warped, "02_warped.jpg", debug_dir)
    else:
        logger.info("Using original image (no quadrilateral contour found)")
        warped = image
        if debug:
            _save_debug(warped, "02_no_warp_fallback.jpg", debug_dir)

    preprocessed = _enhance_for_ocr(warped)

    if debug:
        _save_debug(preprocessed, "03_enhanced.jpg", debug_dir)

    return preprocessed
