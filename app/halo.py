import cv2
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import simpledialog
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import time
import os

np.float = float  # Fix deprecated alias

# Load models
yolo_model = YOLO("yolov8n-seg.pt")
classifier_model = load_model("/Users/isaacigbokwe/Documents/halo/halo/models/halo_multiclass_model.h5")

# Constants
SENSITIVE_CLASSES = {'person', 'cell phone', 'laptop', 'cat', 'dog'}
DETECTION_INTERVAL = 5
NO_BLUR_DURATION_SECONDS = 30
SUSPICIOUS_THRESHOLD = 0.85
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

SUSPICIOUS_CLASSES = {1, 2, 3, 4, 5, 6, 7}  # suspicious labels
CLASS_LABELS = ['normal', 'arrest', 'assault', 'burglary', 'explosion', 'road accident', 'robbery', 'shooting',]

def is_suspicious(model, frames, threshold):
    if len(frames) < 5:
        return False, 0.0, 0  # Not enough data

    resized = [cv2.resize(f, (224, 224)) for f in frames]
    arr = [img_to_array(f) / 255.0 for f in resized]
    x = np.expand_dims(np.stack(arr, axis=0), axis=0)

    probs = model.predict(x, verbose=0)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    is_suspicious = (pred_class in SUSPICIOUS_CLASSES) and (confidence >= threshold)
    return is_suspicious, confidence, pred_class

def blur_masked_region(img, mask, pixel_size=1):
    masked = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    roi = masked[y:y+h, x:x+w]

    if roi.size == 0:
        return img

    small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    tint = np.full_like(small, (106, 211, 255))
    blended = cv2.addWeighted(small, 0.7, tint, 0.3, 0)
    pixelated = cv2.resize(blended, (w, h), interpolation=cv2.INTER_NEAREST)

    pixelated_masked = np.zeros_like(img)
    pixelated_masked[y:y+h, x:x+w] = pixelated
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(img, img, mask=inv_mask)
    combined = cv2.add(background, cv2.bitwise_and(pixelated_masked, pixelated_masked, mask=mask))
    
    return combined

# Maintain a scrolling log of recent predictions
status_log = deque(maxlen=30)  # 30 lines max on the status panel

def draw_status_panel(frame, prediction, probability, class_id):
    status_panel = np.ones((frame.shape[0], 200, 3), dtype=np.uint8) * 255
    label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else "Unknown"
    status_text = f"{label} ({'Suspicious' if prediction else 'Normal'})"

    # Log the current prediction
    log_entry = (status_text, probability)
    status_log.appendleft(log_entry)

    # Draw header
    cv2.putText(status_panel, "HALO STATUS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw scrolling log
    y_start = 60
    line_spacing = 20
    for i, (label, prob) in enumerate(status_log):
        y = y_start + i * line_spacing
        if y + line_spacing > frame.shape[0] - 10:
            break
        fade_intensity = int(min(255, max(0, prob * 255)))
        color = (0, 0, 0 + fade_intensity) if "Suspicious" in label else (0, 0, 0)
        text = f"{label:<20} {prob:.2f}"
        cv2.putText(status_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return np.hstack((frame, status_panel))


def process_video(video_path, output_path, blur_enabled=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width + 200  # for status panel
    out_path = os.path.join(output_path, "output_with_status.mp4")

    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, height))

    no_blur_frames = int(NO_BLUR_DURATION_SECONDS * fps)
    frame_idx, pause_blur_counter = 0, 0
    active_blurs = deque()
    preds = deque(maxlen=VOTE_WINDOW)
    frame_sequence.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Step 1: Copy clean frame for model
        frame_sequence.append(frame.copy())

        # Step 2: Default prediction values
        pred_label, pred_prob, pred_class = False, 0.0, 0

        # Step 3: Evaluate suspicion only every N frames
        if frame_idx % DETECTION_INTERVAL == 0:
            pred_label, pred_prob, pred_class = is_suspicious(classifier_model, frame_sequence, SUSPICIOUS_THRESHOLD)
            preds.append(pred_label)
            if sum(preds) >= VOTE_REQUIREMENT:
                pause_blur_counter = no_blur_frames
        else:
        # Use last vote-based label only for UI if not evaluating
            if preds:
                pred_label = preds[-1]

        results = yolo_model(frame, conf=0.1, verbose=False)[0]

        masks = []
        for mask_tensor, cls_id in zip(results.masks.data, results.boxes.cls):
            class_name = yolo_model.model.names[int(cls_id)]
            if class_name in SENSITIVE_CLASSES:
                raw_mask = mask_tensor.cpu().numpy()
                resized_mask = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                masks.append(binary_mask)

        if blur_enabled and pause_blur_counter <= 0:
            for mask in masks:
                active_blurs.append((mask, 9))
            next_blurs = deque()
            for mask, ttl in active_blurs:
                frame = blur_masked_region(frame, mask)
                if ttl > 1:
                    next_blurs.append((mask, ttl - 1))
            active_blurs = next_blurs
        elif blur_enabled:
            pause_blur_counter -= 1

        combined_frame = draw_status_panel(frame, pred_label, pred_prob, pred_class)
        out_writer.write(combined_frame)
        cv2.imshow("Halo CCTV Feed", combined_frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/isaacigbokwe/Documents/halo/halo/app/input_videos/input5.mp4"
    output_dir = "/Users/isaacigbokwe/Documents/output"

    print("[INFO] Starting video with privacy protection...")
    process_video(video_path, output_dir, blur_enabled=True)

    password = get_password()
    if password == BLUR_OVERRIDE_KEY:
        print("[INFO] Override granted. Replaying without HALO.")
        process_video(video_path, output_dir, blur_enabled=False)
    else:
        print("[INFO] Override denied. Exiting.")
