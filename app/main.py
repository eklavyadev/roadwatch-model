from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import pytesseract
import os
import re
import uuid
import json
import math
import asyncio
import threading
import logging
import time
import numpy as np
from ultralytics import YOLO

# ================= LOGGING ================= #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("roadwatch")

# ================= APP ================= #

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= CONFIG ================= #

MODEL_PATH = "best.pt"
CONF_THRES = 0.25
DEDUP_DISTANCE_M = 20.0

TEMP_DIR = "app/temp_videos"
OUTPUT_DIR = "app/output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log.info("Loading YOLO model from %s …", MODEL_PATH)
model = YOLO(MODEL_PATH)
log.info("YOLO model loaded ✓")

# ================= TASK STORE ================= #
# task_id -> { status, progress, total_frames, potholes_found, result, error }

tasks: dict[str, dict] = {}

# ================= OCR ================= #

def clean_number(s: str) -> str:
    s = re.sub(r"[^0-9\.-]", "", s)
    if s.count("-") > 1:
        s = s.replace("-", "", s.count("-") - 1)
    if "-" in s[1:]:
        s = s.replace("-", ".")
    return s


def extract_lat_lon(frame):
    """Read GPS coordinates from the bottom 25% of the frame via OCR."""
    h, w, _ = frame.shape
    crop = frame[int(h * 0.75):h, 0:w]

    # Upscale 2x — bigger digits reduce 0/8 confusion without breaking OCR
    ch, cw = crop.shape[:2]
    crop = cv2.resize(crop, (cw * 2, ch * 2), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # OTSU auto-selects the best threshold per frame — more robust than fixed 150
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,:-LatLon"
    )

    lat_match = re.search(r"Lat[:\s]*([0-9.,\-]+)", text)
    lon_match = re.search(r"(Lon|Lng)[:\s]*([0-9.,\-]+)", text)

    try:
        if lat_match and lon_match:
            lat = abs(float(clean_number(lat_match.group(1))))
            lon = abs(float(clean_number(lon_match.group(2))))

            # India bounding box — rejects OCR garbage like 0.26 or 260.8
            if not (6.0 <= lat <= 38.0):
                return None, None
            if not (68.0 <= lon <= 98.0):
                return None, None

            return lat, lon
    except (ValueError, IndexError):
        pass

    return None, None


# ================= HELPERS ================= #

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def best_box(boxes):
    """Return the single highest-confidence box from a YOLO Boxes collection."""
    return max(boxes, key=lambda b: float(b.conf[0]))


# ================= BACKGROUND PROCESSING ================= #

def process_video(task_id: str, temp_path: str):
    """Runs in a background thread. Updates tasks[task_id] as it goes."""
    start_time = time.time()
    log.info("[%s] Processing started  →  %s", task_id[:8], os.path.basename(temp_path))

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            log.error("[%s] Could not open video file", task_id[:8])
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = "Could not open video"
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        ocr_y      = int(h * 0.75)
        duration_s = round(total / fps, 1)

        log.info(
            "[%s] Video info  →  %dx%d  |  %.1f fps  |  %d frames  |  ~%ss",
            task_id[:8], w, h, fps, total, duration_s
        )

        tasks[task_id]["total_frames"] = total

        output_path = os.path.join(OUTPUT_DIR, f"output_{task_id}.mp4")
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        candidate_potholes: list[dict] = []
        frame_idx = 0
        ocr_hits = 0
        filtered_boxes = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            time_sec = round(frame_idx / fps, 2)

            # Mask GPS overlay so YOLO never sees it
            yolo_frame = frame.copy()
            cv2.rectangle(yolo_frame, (0, ocr_y), (w, h), (0, 0, 0), cv2.FILLED)

            results = model.predict(
                yolo_frame,
                conf=CONF_THRES,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            annotated = frame.copy()

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Reject false positives that span too much of the frame
                # Real potholes don't cover more than 60% of width OR height
                if (x2 - x1) > w * 0.6 or (y2 - y1) > h * 0.6:
                    filtered_boxes += 1
                    continue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated, f"POTHOLE {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            lat, lon = extract_lat_lon(frame)

            if lat is not None:
                ocr_hits += 1
                cv2.putText(
                    annotated, f"Lat:{lat}  Lon:{lon}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Only consider boxes that passed the size filter
            valid_boxes = [
                b for b in results[0].boxes
                if not ((int(b.xyxy[0][2]) - int(b.xyxy[0][0])) > w * 0.6
                        or  (int(b.xyxy[0][3]) - int(b.xyxy[0][1])) > h * 0.6)
            ]

            if valid_boxes and lat is not None and lon is not None:
                top = best_box(valid_boxes)
                top_conf = float(top.conf[0])
                x1, y1, x2, y2 = map(int, top.xyxy[0])

                matched_idx = None
                for i, candidate in enumerate(candidate_potholes):
                    dist = haversine_distance(lat, lon, candidate["latitude"], candidate["longitude"])
                    if dist <= DEDUP_DISTANCE_M:
                        matched_idx = i
                        break

                record = {
                    "latitude":   lat,
                    "longitude":  lon,
                    "confidence": round(top_conf, 4),
                    "time_sec":   time_sec,
                    "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }

                if matched_idx is None:
                    candidate_potholes.append(record)
                    log.info(
                        "[%s] New pothole #%d  →  lat=%.5f  lon=%.5f  conf=%.2f  t=%.1fs",
                        task_id[:8], len(candidate_potholes), lat, lon, top_conf, time_sec
                    )
                elif top_conf > candidate_potholes[matched_idx]["confidence"]:
                    log.info(
                        "[%s] Updated pothole #%d  →  conf %.2f → %.2f",
                        task_id[:8], matched_idx + 1,
                        candidate_potholes[matched_idx]["confidence"], top_conf
                    )
                    candidate_potholes[matched_idx] = record

            writer.write(annotated)

            # Update progress
            tasks[task_id]["progress"]       = frame_idx
            tasks[task_id]["potholes_found"] = len(candidate_potholes)

            # Log progress every 10%
            pct = round(frame_idx / total * 100)
            if pct % 10 == 0 and frame_idx % max(1, total // 10) < 2:
                log.info("[%s] Progress  →  %d%%  (%d/%d frames)", task_id[:8], pct, frame_idx, total)

        cap.release()
        writer.release()
        os.remove(temp_path)
        log.info("[%s] Temp file deleted  →  %s", task_id[:8], os.path.basename(temp_path))

        potholes = [{"id": i + 1, **p} for i, p in enumerate(candidate_potholes)]
        result = {
            "total_frames_processed": frame_idx,
            "total_potholes_found":   len(potholes),
            "potholes":               potholes,
            "output_video":           output_path,
        }

        json_path = output_path.replace(".mp4", ".json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info("[%s] JSON report saved  →  %s", task_id[:8], json_path)

        result["json_file"] = json_path
        tasks[task_id]["result"] = result
        tasks[task_id]["status"] = "done"

        elapsed = round(time.time() - start_time, 1)
        log.info(
            "[%s] ✅ Done  →  %d frames  |  %d potholes  |  %d OCR hits  |  %d boxes filtered  |  %.1fs elapsed",
            task_id[:8], frame_idx, len(potholes), ocr_hits, filtered_boxes, elapsed
        )

    except Exception as e:
        log.exception("[%s] ❌ Unexpected error: %s", task_id[:8], str(e))
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"]  = str(e)


# ================= API ================= #

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Upload a video. Returns a task_id immediately — poll /progress/{task_id} for updates."""

    task_id   = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{task_id}_{file.filename}")

    log.info("📥 Upload received  →  filename='%s'  task_id=%s", file.filename, task_id[:8])

    content = await file.read()
    size_mb  = round(len(content) / (1024 * 1024), 2)

    with open(temp_path, "wb") as f:
        f.write(content)

    log.info("💾 Buffered to disk  →  %s  (%.2f MB)", os.path.basename(temp_path), size_mb)

    tasks[task_id] = {
        "status":         "processing",
        "progress":       0,
        "total_frames":   0,
        "potholes_found": 0,
        "result":         None,
        "error":          None,
    }

    thread = threading.Thread(target=process_video, args=(task_id, temp_path), daemon=True)
    thread.start()
    log.info("🚀 Processing thread started  →  task_id=%s", task_id[:8])

    return {"task_id": task_id}


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    """
    Server-Sent Events stream.
    Emits a JSON event every 500 ms until processing is done or errored.
    """
    if task_id not in tasks:
        log.warning("Progress requested for unknown task_id=%s", task_id[:8])
        raise HTTPException(status_code=404, detail="Task not found")

    log.info("📡 SSE stream opened  →  task_id=%s", task_id[:8])

    async def event_stream():
        while True:
            task    = tasks[task_id]
            total   = task["total_frames"] or 1
            percent = round(task["progress"] / total * 100)

            if task["status"] == "done":
                payload = {
                    "status":         "done",
                    "percent":        100,
                    "potholes_found": task["potholes_found"],
                    "result":         task["result"],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                log.info("📡 SSE stream closed (done)  →  task_id=%s", task_id[:8])
                break

            if task["status"] == "error":
                payload = {"status": "error", "error": task["error"]}
                yield f"data: {json.dumps(payload)}\n\n"
                log.error("📡 SSE stream closed (error)  →  task_id=%s  error=%s", task_id[:8], task["error"])
                break

            payload = {
                "status":         "processing",
                "percent":        percent,
                "progress":       task["progress"],
                "total_frames":   total,
                "potholes_found": task["potholes_found"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":         "no-cache",
            "X-Accel-Buffering":     "no",
            "X-Content-Type-Options": "nosniff",
            "Transfer-Encoding":     "chunked",
        },
    )
