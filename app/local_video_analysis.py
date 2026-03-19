import cv2
import pytesseract
import requests
import re
import numpy as np
import os

# 🔧 IMPORTANT: Windows tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

YOLO_API = "http://127.0.0.1:8000/detect"  # use LOCAL FastAPI while testing


# ---------------- FRAME EXTRACTION ---------------- #

def extract_frames(video_path, every_n_seconds=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        raise RuntimeError("❌ Could not read FPS from video")

    interval = int(fps * every_n_seconds)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frames.append((count, frame))

        count += 1

    cap.release()
    print(f"🎞 Extracted {len(frames)} frames")
    return frames


# ---------------- OCR ---------------- #

def extract_gps_text(frame):
    h, w, _ = frame.shape

    # bottom 20% crop
    gps_crop = frame[int(h * 0.80):h, 0:w]

    text = pytesseract.image_to_string(
        gps_crop,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,:-LatLon"
    )

    return text


def parse_lat_lon(text):
    lat = lon = None

    lat_match = re.search(r"Lat[:\s]*([0-9.+-]+)", text)
    lon_match = re.search(r"(Lon|Lng)[:\s]*([0-9.+-]+)", text)

    if lat_match:
        lat = float(lat_match.group(1))
    if lon_match:
        lon = float(lon_match.group(2))

    return lat, lon


# ---------------- YOLO ---------------- #

def send_to_yolo(frame):
    # 🚀 resize BEFORE sending (huge speedup)
    frame = cv2.resize(frame, (640, 360))

    ok, img_encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None

    files = {
        "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    res = requests.post(YOLO_API, files=files, timeout=30)

    if res.status_code != 200:
        print("❌ YOLO API error:", res.text)
        return None

    return res.json()


# ---------------- MAIN PIPELINE ---------------- #

def process_video(video_path):
    frames = extract_frames(video_path, every_n_seconds=5)
    results = []

    for idx, frame in frames:
        print(f"\n🔍 Processing frame {idx}")

        # OCR
        text = extract_gps_text(frame)
        lat, lon = parse_lat_lon(text)

        print("📍 OCR text:", text.strip())
        print("📍 Parsed GPS:", lat, lon)

        # YOLO
        yolo_res = send_to_yolo(frame)
        if not yolo_res:
            continue

        detections = yolo_res.get("detections", [])
        print(f"🧠 YOLO detections: {len(detections)}")

        # IMPORTANT: allow detections even if GPS missing (debug mode)
        if detections:
            results.append({
                "frame": idx,
                "latitude": lat,
                "longitude": lon,
                "detections": detections
            })

            print(f"✅ DETECTED object(s) at frame {idx}")

    return results


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    video_path = "road_video_with_gps.mp4"  # <-- make sure path is correct

    if not os.path.exists(video_path):
        raise RuntimeError("❌ Video file not found")

    output = process_video(video_path)

    print("\n================ FINAL OUTPUT ================")
    print(output)
