from ultralytics import YOLO
import cv2

model = YOLO("best.pt")   # 👈 SAME PATH
img = cv2.imread("car.jpg")

results = model.predict(img, conf=0.25, imgsz=640)
print("Boxes:", len(results[0].boxes))

results[0].show()
