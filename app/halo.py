import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Replace with custom model if needed

# Class names to blur (exact string matching from YOLO's class labels)
SENSITIVE_CLASS_NAMES = {'person', 'cell phone', 'laptop', 'cat', 'dog', 'horse'}

# Blur function
def blur_region(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = yolo_model(frame, verbose=False)[0]
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            class_name = yolo_model.model.names[int(cls_id)]
            if class_name in SENSITIVE_CLASS_NAMES:
                frame = blur_region(frame, box)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")

if __name__ == '__main__':
    input_video = "cctv_input.mp4"      # Replace with your input file
    output_video = "cctv_blurred.mp4"   # Desired output file
    process_video(input_video, output_video)
