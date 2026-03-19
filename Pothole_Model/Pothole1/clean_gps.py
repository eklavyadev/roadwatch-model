import os

def filter_unique_potholes(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find '{input_file}'. Run the detector first!")
        return

    print(f"🧹 Cleaning {input_file}...")
    
    # This set will store "Latitude+Longitude" combinations we have already seen
    seen_locations = set()
    unique_lines = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Preserve the header lines (usually the first 2 lines)
    if len(lines) > 0:
        unique_lines.append(lines[0]) # Header
        unique_lines.append(lines[1]) # Separator line

    # Process the data lines (start from index 2)
    # We loop through every line, extract the coordinates, and check for duplicates
    count_duplicates = 0
    
    for line in lines[2:]:
        parts = line.split('|')
        
        # Safety check: ensure line has enough parts (Timestamp, Lat, Long)
        if len(parts) < 3:
            continue

        # Extract just the coordinates (strip spaces to be safe)
        # parts[0] is Timestamp, parts[1] is Lat, parts[2] is Long
        lat = parts[1].strip()
        lng = parts[2].strip()
        
        # Create a unique key for this location
        location_key = (lat, lng)

        # 1. Skip if it's "Unknown" (failed OCR)
        if lat == "Unknown" or lng == "Unknown":
            continue

        # 2. Check if we have seen this exact spot before
        if location_key not in seen_locations:
            seen_locations.add(location_key)
            unique_lines.append(line)
        else:
            count_duplicates += 1

    # Write the clean data to the new file
    with open(output_file, 'w') as f:
        f.writelines(unique_lines)

    print(f"✅ Done!")
    print(f"Original entries: {len(lines) - 2}")
    print(f"Duplicates removed: {count_duplicates}")
    print(f"Unique potholes found: {len(unique_lines) - 2}")
    print(f"📁 Saved clean list to: {output_file}")

# --- Run Configuration ---
if __name__ == "__main__":
    INPUT_FILE = "pothole_coordinates.txt"   # The file created by the previous code
    OUTPUT_FILE = "final_unique_potholes.txt" # The new clean file
    
    filter_unique_potholes(INPUT_FILE, OUTPUT_FILE)