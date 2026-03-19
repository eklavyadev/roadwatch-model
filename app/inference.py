import cv2
import numpy as np
import onnxruntime as ort
import time


def apply_nms(boxes, scores, score_threshold, iou_threshold):
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold,
        iou_threshold
    )

    if indices is None or len(indices) == 0:
        return []

    # Handle OpenCV version differences
    if isinstance(indices[0], (list, tuple, np.ndarray)):
        return [int(i[0]) for i in indices]
    else:
        return [int(i) for i in indices]


class YOLODetector:
    def __init__(self, model_path, conf_thres=0.6, iou_thres=0.5):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = 320

    def preprocess(self, image):
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_shape):
        preds = outputs[0][0]
        img_h, img_w = orig_shape

        raw_boxes = []
        raw_scores = []

        for det in preds:
            obj_conf = det[4]
            if obj_conf < self.conf_thres:
                continue

            class_score = np.max(det[5:])
            score = obj_conf * class_score

            if score < self.conf_thres:
                continue

            cx, cy, bw, bh = det[:4]

            x = int((cx - bw / 2) * img_w)
            y = int((cy - bh / 2) * img_h)
            w = int(bw * img_w)
            h = int(bh * img_h)

            # Road region only
            if y < img_h * 0.4:
                continue

            raw_boxes.append([x, y, w, h])
            raw_scores.append(float(score))

        keep = apply_nms(
            raw_boxes,
            raw_scores,
            self.conf_thres,
            self.iou_thres
        )

        detections = []
        for i in keep:
            detections.append({
                "label": "pothole",
                "confidence": round(raw_scores[i], 3),
                "bbox": raw_boxes[i]
            })

        return detections

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        inp = self.preprocess(image)

        start = time.time()
        outputs = self.session.run(None, {self.input_name: inp})
        infer_time = (time.time() - start) * 1000

        detections = self.postprocess(outputs, (orig_h, orig_w))
        return detections, infer_time
