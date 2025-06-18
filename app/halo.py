import cv2
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import simpledialog
np.float = float  # Fix for deprecated np.float usage
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load models
yolo_model = YOLO("yolov8n.pt")
classifier_model = load_model("/Users/isaacigbokwe/Documents/halo/halo/models/halo_binary_model.h5")

# Constants
SENSITIVE_CLASSES = {'person', 'cell phone', 'laptop', 'cat', 'dog'}
DETECTION_INTERVAL = 10
NO_BLUR_DURATION_SECONDS = 30
SUSPICIOUS_THRESHOLD = 0.95
VOTE_WINDOW = 3
VOTE_REQUIREMENT = 2
BLUR_OVERRIDE_KEY = "hal0hal0"

# Global flags and buffers
blur_override = False
frame_sequence = deque(maxlen=5)
message_timer = 0
message_text = ""

def get_decryption_password():
    root = tk.Tk()
    root.geometry("1x1+200+200")
    root.lift()
    root.attributes("-topmost", True)
    root.after(100, lambda: root.focus_force())
    password = simpledialog.askstring("Access Override", "Enter decryption key:", show='*', parent=root)
    root.destroy()
    time.sleep(0.2)
    return password

def is_suspicious_scene_sequence(model, frame_sequence, threshold=0.7):
    if len(frame_sequence) < 5:
        return False
    frames = [cv2.resize(f, (224, 224)) for f in frame_sequence]
    frames = [img_to_array(f) / 255.0 for f in frames]
    input_tensor = np.expand_dims(np.stack(frames, axis=0), axis=0)
    probability = model.predict(input_tensor, verbose=0)[0][0]
    return probability > threshold

def blur_region(image, bbox, pixel_size=9):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    halo_tint = np.full_like(temp, (106, 211, 255))
    temp = cv2.addWeighted(temp, 0.7, halo_tint, 0.3, 0)
    pixelated = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image

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

def draw_text_box(frame, text, color, x=10, y=20, bg_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)

def process_video(input_path):
    global blur_override, message_text, message_timer

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
        frame_sequence.append(frame)

        if not blur_override and frame_idx % DETECTION_INTERVAL == 0:
            suspicious = is_suspicious_scene_sequence(classifier_model, frame_sequence, threshold=SUSPICIOUS_THRESHOLD)
            prediction_history.append(suspicious)
            if sum(prediction_history) >= VOTE_REQUIREMENT:
                pause_blur_counter = NO_BLUR_DURATION
                print(f"[ALERT] Suspicious activity confirmed at frame {frame_idx}")
            else:
                print(f"Normal activity at frame {frame_idx}")

        results = yolo_model(frame, conf=0.2, verbose=False)[0]
        current_boxes = [box for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)
                         if yolo_model.model.names[int(cls_id)] in SENSITIVE_CLASSES]
        box_history.append(current_boxes)

        if not blur_override and pause_blur_counter <= 0:
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

        # Status indicator
        if blur_override:
            status_text = "HALO DISABLED: DECRYPTION OVERRIDE"
        elif pause_blur_counter > 0:
            status_text = "HALO REMOVED: SUSPICIOUS ACTIVITY DETECTED"
        else:
            status_text = "HALO ACTIVE: PROTECTING PRIVACY"

        status_color = (0, 0, 255) if pause_blur_counter > 0 or blur_override else (0, 255, 0)
        draw_text_box(frame, status_text, status_color)

        # Temporary message display (e.g. access granted)
        if message_timer > 0:
            draw_text_box(frame, message_text, (0, 0, 0), x=10, y=50, bg_color=(255, 255, 255))
            message_timer -= 1

        # Show frame
        cv2.imshow("Halo CCTV Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') and not blur_override:
            password = get_decryption_password()
            if password == BLUR_OVERRIDE_KEY:
                blur_override = True
                message_text = "Access granted"
                print("[AUTH OVERRIDE] Halo disabled by ID:ALPHA1.")
            else:
                message_text = "Invalid key"
                print("[ACCESS DENIED] Incorrect Key - Attempt logged")
            message_timer = int(fps * 2)  # show for 2 seconds

    cap.release()
    cv2.destroyAllWindows()
    print("Video playback finished.")

if __name__ == '__main__':
    input_video = "/Users/isaacigbokwe/Documents/halo/halo/app/input_videos/input2.mp4"
    process_video(input_video)
