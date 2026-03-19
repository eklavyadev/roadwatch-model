import cv2
import torch
from ultralytics import YOLO

# ================= CONFIG ================= #
MODEL_PATH = "best.pt"        # your trained pothole model
VIDEO_PATH = "input.mp4"      # input video
OUTPUT_PATH = "output.mp4"    # output video

CONF_THRES = 0.3              # good for potholes
# ========================================= #

# --------- DEVICE SELECTION (SAFE) -------- #
if torch.cuda.is_available():
    DEVICE = 0
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("⚠️ CUDA not available, using CPU")

# ------------------------------------------ #

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print("🚀 Starting pothole detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False
    )

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"pothole {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("✅ Done! Output saved to:", OUTPUT_PATH)
