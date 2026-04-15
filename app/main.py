from fastapi import FastAPI, UploadFile, File
import cv2
import pytesseract
import os
import re
import uuid
import json
import math
from ultralytics import YOLO

# ================= APP ================= #

app = FastAPI(title="Pothole Detection API")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= CONFIG ================= #

MODEL_PATH = "best.pt"
CONF_THRES = 0.25

# Two detections within this distance (meters) are the same physical pothole.
# Only the highest-confidence one is kept.
DEDUP_DISTANCE_M = 20.0

TEMP_DIR = "app/temp_videos"
OUTPUT_DIR = "app/output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

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
            return (
                float(clean_number(lat_match.group(1))),
                float(clean_number(lon_match.group(2)))
            )
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
    top = max(boxes, key=lambda b: float(b.conf[0]))
    return top


# ================= API ================= #

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    # ---------- Save upload ----------
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        return {"error": "Could not open video"}

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Y-coordinate that separates road view (top) from GPS overlay (bottom 25 %)
    ocr_y = int(h * 0.75)

    output_path = os.path.join(OUTPUT_DIR, f"output_{uuid.uuid4()}.mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # candidate_potholes: best detection seen so far per unique GPS location
    # { latitude, longitude, confidence, time_sec, bbox:{x1,y1,x2,y2} }
    candidate_potholes: list[dict] = []
    frame_idx = 0

    # ---------- FRAME LOOP ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_sec = round(frame_idx / fps, 2)

        # Mask the GPS text region so YOLO never mistakes it for a pothole
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

        # Draw every detected box on the annotated frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated, f"POTHOLE {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        # OCR — run once per frame, reuse for both logging and display
        lat, lon = extract_lat_lon(frame)

        if lat is not None:
            cv2.putText(
                annotated, f"Lat:{lat}  Lon:{lon}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # ---------- Best-pothole logic ----------
        if len(results[0].boxes) > 0 and lat is not None and lon is not None:
            top = best_box(results[0].boxes)
            top_conf = float(top.conf[0])
            x1, y1, x2, y2 = map(int, top.xyxy[0])

            # Look for an existing candidate at the same GPS location
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
                # Genuinely new pothole location
                candidate_potholes.append(record)
            elif top_conf > candidate_potholes[matched_idx]["confidence"]:
                # Better shot of the same pothole — upgrade
                candidate_potholes[matched_idx] = record

        writer.write(annotated)

    cap.release()
    writer.release()
    os.remove(temp_path)

    # ---------- Finalise output ----------
    potholes = [{"id": i + 1, **p} for i, p in enumerate(candidate_potholes)]

    result = {
        "total_frames_processed": frame_idx,
        "total_potholes_found":   len(potholes),
        "potholes": potholes,
        "output_video": output_path,
    }

    # Persist JSON alongside the video
    json_path = output_path.replace(".mp4", ".json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    result["json_file"] = json_path
    return result
