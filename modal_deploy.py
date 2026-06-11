import modal

# 1. Define the environment (Image)
# We start with a Debian slim image, install python requirements from the file,
# install system requirements for OpenCV and Tesseract,
# and add our local source code and model weights.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("tesseract-ocr", "libgl1-mesa-glx", "libglib2.0-0")
    .add_local_file("app/main.py", remote_path="/root/app/main.py")
    .add_local_file("best.pt", remote_path="/root/best.pt")
)

# 2. Define the App
app = modal.App("pothole-detection-service", image=image)

# 3. Define the FastAPI integration
@app.function(
    timeout=600, # Allow up to 10 minutes for video processing (increase if needed)
    min_containers=1, # Keep one container ready
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    # Import main here so it runs within the Modal container environment
    from app.main import app as web_app
    return web_app
