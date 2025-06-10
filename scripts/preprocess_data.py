import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# Directories
normal_dir = '/Users/isaacigbokwe/Documents/halo/materials/clips/normal'
suspicious_dir = '/Users/isaacigbokwe/Documents/halo/materials/clips/suspicious'
output_dir = '/Users/isaacigbokwe/Documents/halo/halo/data/processed_data'
os.makedirs(output_dir, exist_ok=True)

# Parameters
num_frames = 20
frame_size = (224, 224)

# Save CSV label rows as we go
label_data = []

# Helper to load a clip and extract frames, then save individually
clip_index = 0

def process_video(file_path, label):
    global clip_index
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            frame = cv2.resize(frame, frame_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    if len(frames) == num_frames:
        clip_array = np.array(frames)
        clip_filename = f"clip_{clip_index}.npy"
        np.save(os.path.join(output_dir, clip_filename), clip_array)
        label_data.append({'filename': clip_filename, 'label': label})
        clip_index += 1

# Process normal clips
for file in glob(f"{normal_dir}/*.mp4"):
    process_video(file, 0)

# Process suspicious clips
for file in glob(f"{suspicious_dir}/*.mp4"):
    process_video(file, 1)

# Save labels CSV
label_df = pd.DataFrame(label_data)
label_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
print("✅ Incremental data preprocessing complete. Clips saved to processed_data/")