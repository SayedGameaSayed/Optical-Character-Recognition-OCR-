# Egyptian ID OCR

Production-ready OCR pipeline for Egyptian National ID cards.

Extracts full name (Arabic), address (Arabic), and 14-digit National ID number from ID card images with high accuracy.

## Architecture

```
app/
├── main.py                 # FastAPI entry point
├── api/routes.py           # POST /api/v1/extract endpoint
├── core/
│   ├── config.py           # Environment config (Pydantic)
│   └── exceptions.py       # Custom error classes
├── services/
│   ├── preprocessor.py     # OpenCV: perspective warp, denoising, binarization
│   ├── ocr_engine.py       # PaddleOCR wrapper
│   └── validator.py        # Regex, Eastern→Western numeral conversion, noise filtering
└── schemas/models.py       # Pydantic request/response models
```

## Pipeline

1. **Pre-processing** — Edge detection (Canny) → largest quadrilateral contour → perspective warp to 1000×630 px → adaptive binarization → morphological cleanup
2. **OCR** — PaddleOCR (Arabic model) extracts text with bounding boxes and confidence scores
3. **Validation** — Eastern Arabic numerals→Western conversion, 14-digit ID regex, noise keyword filtering, Arabic-script detection

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/egyptian-id-ocr.git
cd egyptian-id-ocr

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@/path/to/id_card.jpg" \
  | jq .
```

### Docker

```bash
docker build -t egyptian-id-ocr .
docker run -p 8000:8000 egyptian-id-ocr
```

## API

### POST `/api/v1/extract`

**Request:** `multipart/form-data` with image file.

**Response:**
```json
{
  "name": "أحمد محمد محمود علي",
  "address": "١٥ شارع التحرير - الدقي - الجيزة",
  "national_id": "29001011234567",
  "confidence_score": 0.8732
}
```

## Testing

```bash
pytest tests/ -v
```

## Deployment

Ready for Railway, Vercel, or any container platform. Set `DEBUG=true` for verbose logging and debug image output.

## License

MIT
