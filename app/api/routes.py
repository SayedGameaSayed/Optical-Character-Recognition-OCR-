from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.exceptions import (
    ImageValidationError,
    NoTextDetectedError,
    OCRProcessingError,
)
from app.schemas.models import ExtractResponse
from app.services.ocr_engine import OCREngine
from app.services.preprocessor import preprocess_image
from app.services.validator import validate_and_parse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])

# Singleton OCR engine (expensive to init per request)
_ocr_engine: OCREngine | None = None


def _get_ocr() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract text from Egyptian National ID card",
    response_description="Structured name, address, and national ID",
)
async def extract(
    file: UploadFile = File(..., description="Image of the ID card"),
) -> ExtractResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    try:
        raw = await file.read()
        if not raw:
            raise ImageValidationError("Empty file")

        # Decode image
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageValidationError("Failed to decode image")

        # Pre-process
        processed = preprocess_image(img)

        # OCR
        ocr = _get_ocr()
        results = ocr.extract(processed)

        if not results:
            raise NoTextDetectedError()

        # Validate and parse
        parsed = validate_and_parse(results)

        return ExtractResponse(**parsed)

    except ImageValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except NoTextDetectedError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except OCRProcessingError as e:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=e.status_code, detail=e.message)
