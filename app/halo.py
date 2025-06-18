import cv2
import numpy as np
from collections import deque
import time
np.float = float  # Fix for deprecated np.float usage
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load models
yolo_model = YOLO("yolov8n.pt")
classifier_model = load_model("halo_binary_model.h5")

# Constants
SENSITIVE_CLASSES = {'person', 'cell phone', 'laptop', 'cat', 'dog'}
DETECTION_INTERVAL = 10                 # Check every N frames
NO_BLUR_DURATION_SECONDS = 5           # Pause blur for this many seconds
SUSPICIOUS_THRESHOLD = 0.7             # CNN probability threshold for "suspicious"
VOTE_WINDOW = 3                        # Voting window size
VOTE_REQUIREMENT = 2                   # Require this many "suspicious" votes

# CNN classification
def is_suspicious_scene(frame, model, target_size=(64, 64), threshold=0.7):
    resized = cv2.resize(frame, target_size)
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probability = model.predict(img_array, verbose=0)[0][0]
    return probability > threshold

# Blur function
def blur_region(image, bbox, pixel_size=9):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return image
    temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    halo_tint = np.full_like(temp, (106, 211, 255))  # BGR
    temp = cv2.addWeighted(temp, 0.7, halo_tint, 0.3, 0)
    pixelated = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image

# IoU-based blur logic
def should_blur(box, history, threshold=0.2, required_matches=3):
    match_count = 0
    box = box.tolist()
    x1b, y1b, x2b, y2b = box
    area1 = (x2b - x1b) * (y2b - y1b)
    most_recent_match = False

    for idx, frame_boxes in enumerate(reversed(history)):
        for past_box in frame_boxes:
            x1a, y1a, x2a, y2a = past_box.tolist()
            xx1 = max(x1a, x1b)
            yy1 = max(y1a, y1b)
            xx2 = min(x2a, x2b)
            yy2 = min(y2a, y2b)
            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter_area == 0:
                continue
            area2 = (x2a - x1a) * (y2a - y1a)
            iou = inter_area / (area1 + area2 - inter_area)
            if iou > threshold:
                match_count += 1
                if idx == 0:
                    most_recent_match = True
                break
    return most_recent_match or match_count >= required_matches

# Main video processing function
def process_video(input_path):
    frame_idx = 0
    pause_blur_counter = 0
    prediction_history = deque(maxlen=VOTE_WINDOW)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    NO_BLUR_DURATION = int(NO_BLUR_DURATION_SECONDS * fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames at {fps:.2f} FPS...")

    box_history = deque(maxlen=33)
    active_blur_boxes = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Classify every DETECTION_INTERVAL frames
        if frame_idx % DETECTION_INTERVAL == 0:
            suspicious = is_suspicious_scene(frame, classifier_model, threshold=SUSPICIOUS_THRESHOLD)
            prediction_history.append(suspicious)
            if sum(prediction_history) >= VOTE_REQUIREMENT:
                pause_blur_counter = NO_BLUR_DURATION
                print(f"[ALERT] Suspicious activity confirmed at frame {frame_idx}")
            else:
                print(f"Normal activity at frame {frame_idx}")

        # Detect objects
        results = yolo_model(frame, conf=0.2, verbose=False)[0]
        current_boxes = [box for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)
                         if yolo_model.model.names[int(cls_id)] in SENSITIVE_CLASSES]
        box_history.append(current_boxes)

        # Apply or skip blurring
        if pause_blur_counter <= 0:
            for box in current_boxes:
                if should_blur(box, box_history):
                    active_blur_boxes.append((box, 6))
            new_active = deque()
            for box, ttl in active_blur_boxes:
                frame = blur_region(frame, box)
                if ttl > 1:
                    new_active.append((box, ttl - 1))
            active_blur_boxes = new_active
        else:
            pause_blur_counter -= 1

        # Status overlay
        status_text = "HALO REMOVED: SUSPICIOUS ACTIVITY DETECTED" if pause_blur_counter > 0 else "HALO ACTIVE"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255) if pause_blur_counter > 0 else (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Halo CCTV Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video playback finished.")

if __name__ == '__main__':
    input_video = "/Users/isaacigbokwe/Documents/halo/halo/app/input_videos/input2.mp4"
    process_video(input_video)
