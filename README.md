# Egyptian ID OCR

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.8-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Production-ready OCR pipeline for Egyptian National ID cards. Extracts **full name (Arabic)**, **address (Arabic)**, and **14-digit National ID number** from card images with robust preprocessing and post-processing.

---

## Architecture

```
app/
├── main.py                   # FastAPI entry point, lifespan, error handlers
├── api/routes.py             # POST /api/v1/extract endpoint
├── core/
│   ├── config.py             # Environment config via Pydantic Settings
│   └── exceptions.py         # Custom exception hierarchy
├── services/
│   ├── preprocessor.py       # OpenCV: edge detection → perspective warp → denoising
│   ├── ocr_engine.py         # PaddleOCR (Arabic) wrapper
│   └── validator.py          # Numeral conversion → regex → noise/reversed-text filtering
└── schemas/models.py         # Pydantic request/response schemas
```

### Pipeline Flow

```
Input Image
    ↓
Pre-processing (Canny edge detection → largest quadrilateral → warp to 1000×630 → fastNlMeans denoising)
    ↓
PaddleOCR (Arabic detection + recognition, det_db_thresh=0.3)
    ↓
Validation (Eastern→Western digits, 14-digit regex, reversed-text fix, noise filtering)
    ↓
Structured JSON: {name, address, national_id, confidence_score}
```

## Edge Case Handling

| Edge Case | Approach |
|---|---|
| **Eastern Arabic numerals** (٠-٩, ۰-۹) | `str.maketrans` maps both U+0660 and U+06F0 ranges to ASCII `[0-9]` |
| **Split 14-digit ID** across OCR regions | All candidates joined → all digit runs extracted → `\d{14}` searched in concatenated string |
| **Reversed/BIDI text artifacts** | `_REVERSED_KEYWORDS` set detected → `text[::-1]` reversal before processing |
| **Noise fields** (ذكر, مسلم, بطاقة, etc.) | ~30 `NOISE_KEYWORDS` filtered from name/address candidates |
| **Non-Arabic text** (English serials, symbols) | Arabic-script gate `[\u0600-\u06FF]` filters non-Arabic lines |
| **No quadrilateral found** | Falls back to original image instead of warp |
| **Empty OCR results** | Returns `{name: "", address: "", national_id: "", confidence_score: 0.0}` |

---

## Quick Start

### Local

```bash
# Clone
git clone https://github.com/SayedGameaSayed/Optical-Character-Recognition-OCR-.git
cd Optical-Character-Recognition-OCR-

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t egyptian-id-ocr .
docker run -p 8000:8000 egyptian-id-ocr
```

---

## API

### `POST /api/v1/extract`

Upload an Egyptian National ID card image and receive structured extraction.

**Request:** `multipart/form-data` with field `file` containing the image (JPEG, PNG).

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@id_card.jpg" \
  | python -m json.tool
```

**Response (200):**

```json
{
  "name": "أحمد محمد محمود",
  "address": "١٥ شارع التحرير - الدقي - الجيزة",
  "national_id": "29001011234567",
  "confidence_score": 0.8732
}
```

**Error Responses:**

| Code | Description |
|---|---|
| `400` | File is not an image, or OCR failed to extract meaningful text |
| `422` | Request body missing or malformed (`multipart/form-data` with `file` required) |
| `500` | Internal server error (check server logs) |

---

## Testing

```bash
# Unit tests (validation logic)
pytest tests/ -v

# End-to-end pipeline test (generates synthetic ID card)
python test_pipeline.py
```

---

## Deployment

Ready for any container platform:

- **Railway** — add `Dockerfile` and set start command
- **Vercel** — use Docker runtime
- **AWS ECS / Google Cloud Run** — push image to registry and deploy
- **Kubernetes** — included health endpoint at `/health`

Set environment variables:

| Variable | Default | Description |
|---|---|---|
| `OCR_CONFIDENCE_THRESHOLD` | `0.3` | Minimum OCR confidence to include candidate |
| `DEBUG` | `False` | Enable debug logging and image output |

---

## License

MIT
