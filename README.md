# RoadWatch Model — Pothole Detection API

A FastAPI service that processes dashcam videos from GPS-enabled recorder apps, detects potholes using a YOLOv8 model, extracts GPS coordinates via OCR, and returns a deduplicated JSON list of pothole locations ready to plot on Google Maps.

---

## How it works

1. **Upload a video** recorded by a geo-tagging dashcam app (GPS coordinates must be overlaid on the bottom 25% of the frame as text, e.g. `Lat: 28.6139 Lon: 77.2090`).
2. **YOLO inference** runs on every frame. The GPS overlay region is masked before inference so the text is never mistaken for a pothole.
3. **OCR** (pytesseract) reads the lat/lon from the bottom 25% of each frame.
4. **Deduplication** — detections within 20 m of each other (Haversine distance) are treated as the same physical pothole. Only the highest-confidence detection per location is kept.
5. **Output** — a JSON response (and a saved `.json` file) with one entry per unique pothole, plus an annotated MP4 with bounding boxes and GPS overlay drawn on every frame.

---

## API

### `POST /analyze-video`

**Request** — multipart form upload:

| Field | Type | Description |
|-------|------|-------------|
| `file` | video file | Dashcam video (MP4, AVI, etc.) |

**Response** — JSON:

```json
{
  "total_frames_processed": 840,
  "total_potholes_found": 3,
  "potholes": [
    {
      "id": 1,
      "latitude": 28.6139,
      "longitude": 77.2090,
      "confidence": 0.873,
      "time_sec": 4.17,
      "bbox": { "x1": 112, "y1": 310, "x2": 264, "y2": 410 }
    }
  ],
  "output_video": "app/output/output_<uuid>.mp4",
  "json_file": "app/output/output_<uuid>.json"
}
```

Use `potholes[].latitude` and `potholes[].longitude` directly as `google.maps.Marker` positions.

---

## Project structure

```
.
├── app/
│   └── main.py        # FastAPI app — inference, OCR, dedup, API endpoint
├── best.pt            # Trained YOLOv8 pothole detection model
├── requirements.txt   # Python dependencies
└── start.sh           # Server startup script
```

Generated at runtime (gitignored):
```
app/output/            # Annotated videos + JSON results
app/temp_videos/       # Temporary upload storage (auto-deleted after processing)
```

---

## Setup

### Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows) — update the path in `app/main.py` if running on Linux/Mac

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server

```bash
bash start.sh
# or directly:
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

The API will be available at `http://localhost:10000`.  
Interactive docs: `http://localhost:10000/docs`

---

## Configuration

These constants at the top of `app/main.py` can be tuned:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONF_THRES` | `0.25` | Minimum YOLO confidence to count as a detection |
| `DEDUP_DISTANCE_M` | `20.0` | Radius in metres — detections closer than this are the same pothole |

---

## Video requirements

- GPS coordinates must be rendered as text in the **bottom 25%** of the frame
- Supported label formats: `Lat: 28.6139 Lon: 77.2090` or `Lat: 28.6139 Lng: 77.2090`
- Works with any standard dashcam GPS overlay app
