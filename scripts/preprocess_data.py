import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# Directories
normal_dir = '/Users/isaacigbokwe/Documents/halo/materials/clips/normal'
suspicious_base_dir = '/Users/isaacigbokwe/Documents/halo/materials/clips/suspicious'
output_dir = '/Users/isaacigbokwe/Documents/halo/halo/data/processed_data'
os.makedirs(output_dir, exist_ok=True)

# Parameters optimized for EfficientNet-B0
num_frames = 5  # Reduced from 20 to 5 for minimum fps requirement
frame_size = (224, 224)  # EfficientNet-B0 native input size

# Suspicious categories and labels
suspicious_categories = {
    'Arrest': 1,
    'Assault': 2,
    'Burglary': 3,
    'Explosion': 4,
    'Road_Accidents': 5,
    'Robbery': 6,
    'Shooting': 7
}

# Save CSV label rows as we go
label_data = []
clip_index = 0

def process_video(file_path, label):
    global clip_index
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"⚠️ Warning: Could not open video file {file_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print(f"⚠️ Warning: Video {file_path} has only {total_frames} frames (needs {num_frames})")
        cap.release()
        return
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Warning: Failed to read frame {idx} from {file_path}")
            break
        if idx in frame_indices:
            try:
                # Convert BGR to RGB for EfficientNet compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, frame_size)
                # EfficientNet-B0 expects values in [0, 1] range (ImageNet normalization applied later)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            except Exception as e:
                print(f"⚠️ Warning: Error processing frame {idx} from {file_path}: {str(e)}")
                break
    
    cap.release()
    
    if len(frames) == num_frames:
        clip_array = np.array(frames, dtype=np.float32)  # Explicit float32 for efficiency
        clip_filename = f"clip_{clip_index}.npy"
        np.save(os.path.join(output_dir, clip_filename), clip_array)
        label_data.append({'filename': clip_filename, 'label': label})
        clip_index += 1
    else:
        print(f"⚠️ Warning: Could only extract {len(frames)} frames from {file_path}")

# Process normal clips
print("Processing normal videos...")
for file in glob(os.path.join(normal_dir, "*.mp4")):
    process_video(file, 0)

# Process suspicious subfolders
print("\nProcessing suspicious videos...")
for category, label in suspicious_categories.items():
    category_path = os.path.join(suspicious_base_dir, category)
    if not os.path.exists(category_path):
        print(f"⚠️ Warning: Category directory {category_path} does not exist")
        continue
        
    print(f"Processing {category} videos (label {label})...")
    for file in glob(os.path.join(category_path, "*.mp4")):
        process_video(file, label)

# Save labels CSV
label_df = pd.DataFrame(label_data)
label_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)

# Verification
print("\n✅ Processing complete. Verification:")
print(f"- Total NPY files created: {len([f for f in os.listdir(output_dir) if f.endswith('.npy')])}")
print(f"- Total labels in CSV: {len(label_df)}")
print(f"- Label distribution:\n{label_df['label'].value_counts()}")