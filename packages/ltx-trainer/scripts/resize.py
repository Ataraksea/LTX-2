import os
import cv2
import numpy as np

def resize_and_pad_video(video_path, output_dir, target_w=1920, target_h=1080):
    """Resizes videos using padding (if smaller) or top-center cropping (if larger)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)

    # Check if the output file already exists to skip reprocessing
    if os.path.exists(output_path):
        print(f"Skipping {video_name}: Output already exists.")
        cap.release()
        return

    # Get original properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    print(f"Processing {video_name} ({orig_w}x{orig_h} -> {target_w}x{target_h})...")

    # Determine processing strategy based on dimensions
    is_smaller = (orig_w <= target_w) and (orig_h <= target_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blank black canvas of the target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if is_smaller:
            # --- STRATEGY 1: Center with Black Padding ---
            # Calculate top-left starting coordinates to center the original frame
            x_offset = (target_w - orig_w) // 2
            y_offset = (target_h - orig_h) // 2
            canvas[y_offset:y_offset+orig_h, x_offset:x_offset+orig_w] = frame
        else:
            # --- STRATEGY 2: Downscale Aspect Ratio + Top-Center Crop ---
            # Determine scale factor based on matching the width or height first
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            scale = max(scale_w, scale_h)  # Scale up/down to cover the entire canvas

            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Perform top-center crop
            x_crop_start = (new_w - target_w) // 2
            y_crop_start = 0  # 0 forces the crop to anchor at the top edge

            canvas = resized_frame[y_crop_start:y_crop_start+target_h, x_crop_start:x_crop_start+target_w]

        out.write(canvas)

    cap.release()
    out.release()
    print(f"Finished resizing {video_name}.")

def process_directory(input_directory, output_directory, target_width, target_height):
    """Iterates through a directory to resize all video files."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(valid_extensions):
            full_input_path = os.path.join(input_directory, filename)
            resize_and_pad_video(full_input_path, output_directory, target_width, target_height)

INPUT_DIR = "D:/Downloads/LTX-2/packages/ltx-trainer/dataset/"
OUTPUT_DIR = "D:/Downloads/LTX-2/packages/ltx-trainer/output/"

# Specify your target dimensions here
TARGET_WIDTH = 416
TARGET_HEIGHT = 704

process_directory(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH, TARGET_HEIGHT)