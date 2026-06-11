from __future__ import annotations


class OCRProcessingError(Exception):
    """Raised when image pre-processing or OCR extraction fails."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ImageValidationError(OCRProcessingError):
    """Raised when the input image is invalid or corrupted."""

    def __init__(self, message: str = "Invalid or corrupted image") -> None:
        super().__init__(message, status_code=400)


class NoTextDetectedError(OCRProcessingError):
    """Raised when OCR returns no text."""

    def __init__(self, message: str = "No text detected in image") -> None:
        super().__init__(message, status_code=422)
