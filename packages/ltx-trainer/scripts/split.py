import os
import cv2

def split_video_by_frames(video_path, output_dir, chunk_size=121):
    """Splits a single video into chunks of exactly `chunk_size` frames, skipping if they exist."""
    # Capture the input video to read properties and calculate total frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Calculate total frames to pre-determine how many chunks should exist
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_chunks = total_frames // chunk_size

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Check if all expected output files already exist
    all_chunks_exist = True
    for chunk_idx in range(expected_chunks):
        expected_filename = f"{video_base_name}_part{chunk_idx:03d}.mp4"
        expected_path = os.path.join(output_dir, expected_filename)
        if not os.path.exists(expected_path):
            all_chunks_exist = False
            break

    # Skip processing if every required chunk is already present
    if expected_chunks > 0 and all_chunks_exist:
        print(f"Skipping {video_base_name}: All {expected_chunks} chunks already exist.")
        cap.release()
        return
    elif expected_chunks == 0:
        print(f"Skipping {video_base_name}: Video is shorter than {chunk_size} frames.")
        cap.release()
        return

    # Gather source video properties for writing
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    chunk_index = 0
    out = None
    frames_in_current_chunk = 0

    print(f"Processing {video_base_name} (Expecting {expected_chunks} chunks)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize a new VideoWriter if we are at the start of a chunk
        if frames_in_current_chunk == 0:
            output_filename = f"{video_base_name}_part{chunk_index:03d}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            chunk_index += 1

        # Write the current frame to the active chunk
        out.write(frame)
        frames_in_current_chunk += 1

        # Close the writer once the exact chunk size is met
        if frames_in_current_chunk == chunk_size:
            out.release()
            out = None
            frames_in_current_chunk = 0

    # Clean up assets
    cap.release()
    if out is not None:
        # Delete trailing partial frames that didn't hit 121
        out.release()
        os.remove(output_path)

    print(f"Finished {video_base_name}.")

def process_directory(input_directory, output_directory):
    """Iterates through a directory to process all video files."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(valid_extensions):
            full_input_path = os.path.join(input_directory, filename)
            split_video_by_frames(full_input_path, output_directory)

# --- Execution ---
INPUT_DIR = "D:/Downloads/LTX-2/packages/ltx-trainer/dataset/0.67"
OUTPUT_DIR = "D:/Downloads/LTX-2/packages/ltx-trainer/output/0.67"

process_directory(INPUT_DIR, OUTPUT_DIR)

