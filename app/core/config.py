from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Egyptian ID OCR"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_image_size_mb: int = 10
    paddleocr_lang: str = "ar"
    paddleocr_use_angle_cls: bool = True
    ocr_confidence_threshold: float = 0.3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
