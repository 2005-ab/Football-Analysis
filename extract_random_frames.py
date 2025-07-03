import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def extract_continuous_frames(video_path, output_dir, num_frames=100, margin_ratio=0.1):
    """
    Extract a continuous block of frames from a random position in the middle of the video.
    Avoids the first and last margin_ratio of the video.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting {num_frames} continuous frames from the middle...")
    
    # Calculate safe margins
    margin = int(total_frames * margin_ratio)
    min_start = margin
    max_start = total_frames - margin - num_frames
    if max_start <= min_start:
        print("Video too short for requested extraction.")
        return []
    start_idx = random.randint(min_start, max_start)
    frame_indices = list(range(start_idx, start_idx + num_frames))
    
    extracted_frames = []
    with tqdm(total=len(frame_indices), desc="Extracting frames") as pbar:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame)
                pbar.update(1)
            else:
                print(f"Warning: Could not read frame {frame_idx}")
    cap.release()
    print(f"✅ Extracted {len(extracted_frames)} frames to {output_dir}")
    return extracted_frames

def create_video_from_frames(frames, output_path, fps=24):
    """
    Create a video from extracted frames
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    if not frames:
        print("No frames to create video from")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"Creating video from {len(frames)} frames...")
    with tqdm(total=len(frames), desc="Writing video") as pbar:
        for frame in frames:
            out.write(frame)
            pbar.update(1)
    
    out.release()
    print(f"✅ Video saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    video_path = r"D:\strategy\full_match.mp4"
    output_dir = "extracted_frames"
    output_video = "random_frames_sample.avi"
    
    # Extract random frames
    frames = extract_continuous_frames(video_path, output_dir, num_frames=100)
    
    # Create a video from the extracted frames (optional)
    if frames:
        create_video_from_frames(frames, output_video, fps=10)
        print(f"\n🎯 Ready for tracking analysis!")
        print(f"   📁 Frames saved in: {output_dir}")
        print(f"   🎬 Sample video: {output_video}")
        print(f"   📊 Total frames: {len(frames)}")
    else:
        print("❌ No frames were extracted!") 