import cv2
from ultralytics import YOLO
import easyocr
import os
import re
import math

# --- PART 1: HAVERSINE FORMULA ---
# Calculates distance in meters between two GPS points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance # Returns meters

# --- PART 2: THE MAIN PROCESS ---
def process_video_haversine(input_video_path, model_path, output_video_path, log_file_path):
    # Setup
    if not os.path.exists(input_video_path):
        print(f"❌ Error: Video '{input_video_path}' not found.")
        return

    print(f"🧠 Loading YOLO model...")
    model = YOLO(model_path)

    print(f"📖 Loading Text Reader...")
    reader = easyocr.Reader(['en'], gpu=False) 

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- CONFIGURATION (UPDATED) ---
    DISTANCE_THRESHOLD = 50.0  # <--- CHANGED TO 50 METERS
    COOLDOWN_SECONDS = 2.0     # Wait 2 seconds before scanning again
    COOLDOWN_FRAMES = int(fps * COOLDOWN_SECONDS)
    
    # State variables
    last_logged_coords = None # To store (lat, long) of the last verified pothole
    pothole_cooldown = 0 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Single Output File
    with open(log_file_path, "w") as f:
        f.write(f"{'TIMESTAMP':<10} | {'LATITUDE':<20} | {'LONGITUDE':<20} | {'DIST. FROM LAST'}\n")
        f.write("-" * 80 + "\n")

    print(f"▶️ Processing with {DISTANCE_THRESHOLD}m Threshold & {COOLDOWN_SECONDS}s Timer...")

    # Safe Window Check
    show_window = True
    try:
        test_img = cv2.imread(model_path) 
    except:
        pass 

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time_sec = frame_count / fps

        # AI Detection
        results = model.predict(frame, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            if pothole_cooldown == 0:
                print(f"🕳️ Pothole at {current_time_sec:.2f}s! Scanning GPS...")

                # Crop Bottom-Left
                y_start = int(height * 0.75)
                y_end = height
                x_start = 0
                x_end = int(width * 0.60)
                gps_crop = frame[y_start:y_end, x_start:x_end]
                
                # Read Text
                gps_text_list = reader.readtext(gps_crop, detail=0)
                full_text = " ".join(gps_text_list)

                # Extract EXACT Coordinates (No rounding)
                lat_match = re.search(r"Lat\s*([\d\.]+)", full_text)
                long_match = re.search(r"Long\s*([\d\.]+)", full_text)

                if lat_match and long_match:
                    try:
                        current_lat = float(lat_match.group(1))
                        current_long = float(long_match.group(1))
                        
                        # --- HAVERSINE CHECK ---
                        should_log = False
                        dist_msg = "N/A (First)"

                        if last_logged_coords is None:
                            # First pothole ever - always log
                            should_log = True
                        else:
                            # Calculate distance from last logged pothole
                            dist = haversine_distance(
                                last_logged_coords[0], last_logged_coords[1],
                                current_lat, current_long
                            )
                            dist_msg = f"{dist:.2f}m"
                            
                            if dist > DISTANCE_THRESHOLD:
                                should_log = True
                            else:
                                print(f"   ⚠️ Skipped: Too close to previous ({dist:.2f}m < {DISTANCE_THRESHOLD}m)")

                        if should_log:
                            # Write exact values to file
                            log_entry = f"{current_time_sec:.2f}s".ljust(10) + \
                                        f" | {str(current_lat):<20} | {str(current_long):<20} | {dist_msg}"
                            
                            with open(log_file_path, "a") as f:
                                f.write(log_entry + "\n")
                            
                            # Update state
                            last_logged_coords = (current_lat, current_long)
                            print(f"   ✅ Logged New Pothole: {dist_msg} away.")

                            # Draw Green Box (Success)
                            cv2.rectangle(annotated_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                        else:
                            # Draw Orange Box (Detected but Skipped)
                            cv2.rectangle(annotated_frame, (x_start, y_start), (x_end, y_end), (0, 165, 255), 2)

                    except ValueError:
                        print("   ❌ Error parsing GPS numbers.")

                # Reset timer
                pothole_cooldown = COOLDOWN_FRAMES 

        if pothole_cooldown > 0:
            pothole_cooldown -= 1

        out.write(annotated_frame)
        
        if show_window:
            try:
                cv2.imshow('Pothole Tracker', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                show_window = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Finished! Log saved to: {log_file_path}")

if __name__ == "__main__":
    process_video_haversine(
        input_video_path="road_video.mp4",
        model_path="best.pt",
        output_video_path="output_gps_haversine_50m.mp4",
        log_file_path="pothole_coordinates_50m.txt"
    )