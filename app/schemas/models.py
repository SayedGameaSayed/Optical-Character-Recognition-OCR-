from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    """Optional metadata for the extraction request."""

    debug: bool = Field(False, description="Save debug images")
    confidence_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum confidence to include text"
    )


class ExtractResponse(BaseModel):
    """Structured OCR extraction result."""

    name: str = Field("", description="Full name in Arabic")
    address: str = Field("", description="Address in Arabic")
    national_id: str = Field("", description="14-digit National ID number")
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall OCR confidence"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
