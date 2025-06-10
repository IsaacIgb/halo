import os
import csv
import math
import subprocess
import datetime

# === CONFIGURATION ===
input_video_dir = "/Users/isaacigbokwe/Documents/halo/materials/raw_videos"
suspicious_output_dir = "/Users/isaacigbokwe/Documents/halo/materials/clips/suspicious"
normal_output_dir = "/Users/isaacigbokwe/Documents/halo/materials/clips/normal"
csv_path = "/Users/isaacigbokwe/Documents/halo/materials/labeled_video_segments_contextual.csv"

# Create output directories if they don't exist
os.makedirs(suspicious_output_dir, exist_ok=True)
os.makedirs(normal_output_dir, exist_ok=True)

# === HELPERS ===
def time_to_seconds(t):
    """Converts time string to float seconds."""
    try:
        dt = datetime.datetime.strptime(t.strip(), "%H:%M:%S.%f")
    except ValueError:
        try:
            dt = datetime.datetime.strptime(t.strip(), "%M:%S.%f")
        except ValueError:
            dt = datetime.datetime.strptime(t.strip(), "%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def run_ffmpeg_clip(input_path, start, duration, output_path):
    """Calls ffmpeg to generate a video clip."""
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === MAIN SCRIPT ===
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        video_filename = row["video_filename"].strip()
        start_time = time_to_seconds(row["start_time"])
        end_time = time_to_seconds(row["end_time"])
        label = row["label"].strip().lower()

        if label not in {"suspicious", "normal"}:
            print(f"[!] Invalid label '{label}' in row: {row}")
            continue

        input_path = os.path.join(input_video_dir, video_filename)
        if not os.path.exists(input_path):
            print(f"[!] Video not found: {input_path}")
            continue

        output_dir = suspicious_output_dir if label == "suspicious" else normal_output_dir
        segment_duration = end_time - start_time
        num_clips = math.ceil(segment_duration / 5)

        for i in range(num_clips):
            clip_start = start_time + i * 5
            clip_duration = min(5, end_time - clip_start)
            if clip_duration <= 0:
                continue

            base_name = os.path.splitext(video_filename)[0]
            output_filename = f"{base_name}_part{i}_{label}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            run_ffmpeg_clip(input_path, clip_start, clip_duration, output_path)
            print(f"[+] Saved: {output_path}")
