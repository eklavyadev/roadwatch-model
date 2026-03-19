import cv2
import pytesseract
import re

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
    Returns (lat, lon) or (None, None)
    """
    h, w, _ = frame.shape

    # Bottom 25% crop (same as your tested OCR)
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
