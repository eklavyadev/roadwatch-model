import cv2
import pytesseract
import re
import json
from collections import deque
from datetime import datetime

# Windows only (remove on Linux/Mac)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_number(s: str):
    s = re.sub(r"[^0-9\.-]", "", s)
    if s.count("-") > 1:
        s = s.replace("-", "", s.count("-") - 1)
    if "-" in s[1:]:
        s = s.replace("-", ".")
    return s

def extract_lat_lon(frame):
    """
    Extracts OCR strictly from the bottom 25% of the frame.
    """
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
            return lat, lon
    except ValueError:
        pass

    return None, None


def scan_video_for_potholes(video_path, yolo_model, output_json="potholes.json"):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    text_boundary_y = int(height * 0.75)
    trigger_y = text_boundary_y - 15 
    
    recent_logs_x = deque(maxlen=5) 
    database_records = []

    # --- YOUR SPECIFIC REQUIREMENTS ---
    MIN_WIDTH = 40
    MIN_HEIGHT = 20
    MIN_CONFIDENCE = 0.30  # Filter out anything under 30% confidence

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_frame = frame.copy()
        # Black out the OCR text area so YOLO doesn't see it
        cv2.rectangle(yolo_frame, (0, text_boundary_y), (width, height), (0, 0, 0), -1)

        # ---------------------------------------------------------
        # 1. RUN YOLO ON THE MASKED FRAME
        # ---------------------------------------------------------
        # results = yolo_model(yolo_frame)
        # boxes = results[0].boxes.data.cpu().numpy() # Use .data to get confidence scores
        
        boxes = [] # MOCK VARIABLE: Replace with actual YOLO inference
        
        for box in boxes:
            # Unpack the 6 values YOLOv8 provides: [x1, y1, x2, y2, confidence, class_id]
            x_min, y_min, x_max, y_max, conf, class_id = box
            
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # 2. FILTER: Check size AND check if confidence is >= 30%
            if box_width >= MIN_WIDTH and box_height >= MIN_HEIGHT and conf >= MIN_CONFIDENCE:
                
                x_center = (x_min + x_max) / 2
                
                # 3. Check if it hit the trigger line
                if y_max >= trigger_y:
                    is_duplicate = any(abs(x_center - logged_x) < 50 for logged_x in recent_logs_x)
                    
                    if not is_duplicate:
                        
                        lat, lon = extract_lat_lon(frame)
                        
                        if lat is not None and lon is not None:
                            record = {
                                "id": len(database_records) + 1,
                                "latitude": lat,
                                "longitude": lon,
                                "confidence": float(round(conf, 2)), # Save the confidence score to the JSON
                                "pixel_width": float(box_width),
                                "pixel_height": float(box_height),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            print(f"🚨 Logged (Conf: {conf:.2f}): {record}")
                            database_records.append(record)
                            recent_logs_x.append(x_center)
                        
        cv2.line(yolo_frame, (0, int(trigger_y)), (width, int(trigger_y)), (0, 0, 255), 2)
        cv2.imshow("Scanner Debug View", yolo_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # ---------------------------------------------------------
    # EXPORT TO JSON
    # ---------------------------------------------------------
    if database_records:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(database_records, f, indent=4)
        print(f"\n✅ Processing complete. {len(database_records)} potholes saved to '{output_json}'.")
    else:
        print("\n⚠️ Processing complete. No potholes met the requirements to be saved.")
