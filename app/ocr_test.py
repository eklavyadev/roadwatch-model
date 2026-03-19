import cv2
import pytesseract
import os

# 🔧 Windows: tell pytesseract where tesseract.exe is
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

VIDEO_PATH = "road_video_with_gps.mp4"   # 👈 change if needed
EVERY_N_SECONDS = 1                      # try 1 first, then 2 or 3
SAVE_DEBUG_FRAMES = True                 # saves cropped GPS region


def ocr_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        raise RuntimeError("❌ Could not read video FPS")

    interval = int(fps * EVERY_N_SECONDS)
    frame_count = 0
    extracted = 0

    os.makedirs("ocr_debug", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            h, w, _ = frame.shape

            # ✅ Crop bottom 25% (adjust if needed)
            gps_crop = frame[int(h * 0.75):h, 0:w]

            # OCR with whitelist (VERY IMPORTANT)
            text = pytesseract.image_to_string(
                gps_crop,
                config="--psm 6 -c tessedit_char_whitelist=0123456789.,:-LatLon"
            )

            print(f"\n🟡 Frame {frame_count}")
            print("OCR TEXT:")
            print(text.strip())

            # Save crop for visual inspection
            if SAVE_DEBUG_FRAMES:
                cv2.imwrite(
                    f"ocr_debug/frame_{frame_count}.jpg",
                    gps_crop
                )

            extracted += 1

        frame_count += 1

    cap.release()
    print(f"\n✅ Processed {extracted} OCR frames")


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        raise RuntimeError("❌ Video file not found")

    ocr_from_video(VIDEO_PATH)
