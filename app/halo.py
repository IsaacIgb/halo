import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import simpledialog
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import time

np.float = float  # Fix deprecated alias

# Load models
yolo_model = YOLO("yolov8n.pt")
classifier_model = load_model("/Users/isaacigbokwe/Documents/halo/halo/models/halo_binary_model.h5")

# Constants
SENSITIVE_CLASSES = {'person', 'cell phone', 'laptop', 'cat', 'dog'}
DETECTION_INTERVAL = 10
NO_BLUR_DURATION_SECONDS = 30
SUSPICIOUS_THRESHOLD = 0.9
VOTE_WINDOW = 3
VOTE_REQUIREMENT = 2
BLUR_OVERRIDE_KEY = "hal0hal0"

frame_sequence = deque(maxlen=5)

def get_password():
    root = tk.Tk()
    root.withdraw()
    pw = simpledialog.askstring("Override Access", "Enter decryption key:", show='*')
    root.destroy()
    return pw

def is_suspicious(model, frames, threshold):
    if len(frames) < 5:
        return False
    resized = [cv2.resize(f, (224, 224)) for f in frames]
    arr = [img_to_array(f) / 255.0 for f in resized]
    x = np.expand_dims(np.stack(arr, axis=0), axis=0)
    prob = model.predict(x, verbose=0)[0][0]
    return prob > threshold

def blur_region(img, bbox, pixel_size=9):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    small = cv2.resize(roi, (pixel_size, pixel_size))
    tint = np.full_like(small, (106, 211, 255))
    blended = cv2.addWeighted(small, 0.7, tint, 0.3, 0)
    pixelated = cv2.resize(blended, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pixelated
    return img

def draw_status(frame, text, y=20, color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (10, y - th - 6), (10 + tw + 10, y + 6), (255, 255, 255), -1)
    cv2.putText(frame, text, (15, y), font, scale, color, thick)

def process_video(video_path, blur_enabled=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    no_blur_frames = int(NO_BLUR_DURATION_SECONDS * fps)
    frame_idx, pause_blur_counter = 0, 0
    box_history = deque(maxlen=66)
    active_blurs = deque()
    preds = deque(maxlen=VOTE_WINDOW)
    frame_sequence.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_sequence.append(frame)

        if blur_enabled and frame_idx % DETECTION_INTERVAL == 0:
            preds.append(is_suspicious(classifier_model, frame_sequence, SUSPICIOUS_THRESHOLD))
            if sum(preds) >= VOTE_REQUIREMENT:
                pause_blur_counter = no_blur_frames

        results = yolo_model(frame, conf=0.2, verbose=False)[0]
        boxes = [b for b, c, i in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)
                 if yolo_model.model.names[int(i)] in SENSITIVE_CLASSES]
        box_history.append(boxes)

        if blur_enabled and pause_blur_counter <= 0:
            for box in boxes:
                active_blurs.append((box, 12))
            next_blurs = deque()
            for box, ttl in active_blurs:
                frame = blur_region(frame, box)
                if ttl > 1:
                    next_blurs.append((box, ttl - 1))
            active_blurs = next_blurs
        elif blur_enabled:
            pause_blur_counter -= 1

        # Status message
        if not blur_enabled:
            draw_status(frame, "HALO DISABLED: DECRYPTION OVERRIDE", color=(0, 0, 255))
        elif pause_blur_counter > 0:
            draw_status(frame, "HALO REMOVED: SUSPICIOUS ACTIVITY DETECTED", color=(0, 0, 255))
        else:
            draw_status(frame, "HALO ACTIVE: PROTECTING PRIVACY", color=(0, 255, 0))

        cv2.imshow("Halo CCTV Feed", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/isaacigbokwe/Documents/halo/halo/app/input_videos/input2.mp4"

    print("[INFO] Starting video with privacy protection...")
    process_video(video_path, blur_enabled=True)

    # Prompt at the end
    password = get_password()
    if password == BLUR_OVERRIDE_KEY:
        print("[INFO] Override granted. Replaying without HALO.")
        process_video(video_path, blur_enabled=False)
    else:
        print("[INFO] Override denied. Exiting.")
