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
from ultralytics import YOLO

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

model = YOLO(MODEL_PATH)

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

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,:-LatLon"
    )

    lat_match = re.search(r"Lat[:\s]*([0-9.,\-]+)", text)
    lon_match = re.search(r"(Lon|Lng)[:\s]*([0-9.,\-]+)", text)

    try:
        if lat_match and lon_match:
            lat = float(clean_number(lat_match.group(1)))
            lon = float(clean_number(lon_match.group(2)))

            # Reject obviously malformed values (dropped decimal point, stray sign, etc.)
            if not (-90.0 <= lat <= 90.0):
                return None, None
            if not (-180.0 <= lon <= 180.0):
                return None, None
            # Zero coords mean OCR produced garbage
            if lat == 0.0 or lon == 0.0:
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
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = "Could not open video"
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        ocr_y      = int(h * 0.75)

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

                # Reject whole-frame false positives — real potholes don't span
                # more than 70% of both the frame width and height simultaneously
                if (x2 - x1) > w * 0.7 and (y2 - y1) > h * 0.7:
                    continue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated, f"POTHOLE {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            lat, lon = extract_lat_lon(frame)

            if lat is not None:
                cv2.putText(
                    annotated, f"Lat:{lat}  Lon:{lon}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Only consider boxes that passed the size filter
            valid_boxes = [
                b for b in results[0].boxes
                if not ((int(b.xyxy[0][2]) - int(b.xyxy[0][0])) > w * 0.7
                        and (int(b.xyxy[0][3]) - int(b.xyxy[0][1])) > h * 0.7)
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
                elif top_conf > candidate_potholes[matched_idx]["confidence"]:
                    candidate_potholes[matched_idx] = record

            writer.write(annotated)

            # Update progress
            tasks[task_id]["progress"]       = frame_idx
            tasks[task_id]["potholes_found"] = len(candidate_potholes)

        cap.release()
        writer.release()
        os.remove(temp_path)

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

        result["json_file"] = json_path
        tasks[task_id]["result"] = result
        tasks[task_id]["status"] = "done"

    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"]  = str(e)


# ================= API ================= #

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Upload a video. Returns a task_id immediately — poll /progress/{task_id} for updates."""

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status":        "processing",
        "progress":      0,
        "total_frames":  0,
        "potholes_found": 0,
        "result":        None,
        "error":         None,
    }

    thread = threading.Thread(target=process_video, args=(task_id, temp_path), daemon=True)
    thread.start()

    return {"task_id": task_id}


@app.get("/progress/{task_id}")
async def progress(task_id: str):
    """
    Server-Sent Events stream.
    Emits a JSON event every 500 ms until processing is done or errored.

    Event shape while processing:
      { status, progress, total_frames, potholes_found, percent }

    Final event on completion:
      { status: "done", result: { ... } }
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        while True:
            task = tasks[task_id]
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
                break

            if task["status"] == "error":
                payload = {"status": "error", "error": task["error"]}
                yield f"data: {json.dumps(payload)}\n\n"
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables Nginx buffering if behind a proxy
        },
    )
