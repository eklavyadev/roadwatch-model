from fastapi import FastAPI, UploadFile, File
import cv2
import pytesseract
import os
import re
import uuid
from ultralytics import YOLO

# ================= APP ================= #

app = FastAPI(title="Pothole Detection API")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= CONFIG ================= #

MODEL_PATH = "best.pt"
CONF_THRES = 0.25

TEMP_DIR = "app/temp_videos"
OUTPUT_DIR = "app/output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# ================= OCR ================= #

def clean_number(s):
    s = re.sub(r"[^0-9\.-]", "", s)
    if s.count("-") > 1:
        s = s.replace("-", "", s.count("-") - 1)
    if "-" in s[1:]:
        s = s.replace("-", ".")
    return s

def extract_lat_lon(frame):
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
    except:
        pass

    return None, None

# ================= API ================= #

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    # ---------- Save upload ----------
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(
        OUTPUT_DIR,
        f"output_{uuid.uuid4()}.mp4"
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    detections_log = []
    frame_idx = 0

    # ---------- FULL VIDEO LOOP ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_sec = round(frame_idx / fps, 2)

        results = model.predict(
            frame,
            conf=CONF_THRES,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        annotated = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"POTHOLE {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        writer.write(annotated)


        lat, lon = extract_lat_lon(frame)

        if lat and lon:
            cv2.putText(
                annotated,
                f"Lat:{lat} Lon:{lon}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if len(results[0].boxes) > 0 and lat and lon:
            detections_log.append({
                "time_sec": time_sec,
                "latitude": lat,
                "longitude": lon,
                "count": len(results[0].boxes)
            })

        

    cap.release()
    writer.release()
    os.remove(temp_path)

    return {
        "total_frames_processed": frame_idx,
        "detections": detections_log,
        "output_video": output_path
    }
