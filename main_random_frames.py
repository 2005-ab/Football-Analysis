from trackers.tracker import Tracker
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob

def load_extracted_frames(frames_dir):
    """Load extracted frames from directory"""
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    frames = []
    
    print(f"🔄 Loading {len(frame_files)} extracted frames from {frames_dir}")
    with tqdm(total=len(frame_files), desc="Loading frames") as pbar:
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                frames.append(frame)
            pbar.update(1)
    
    print(f"✅ Loaded {len(frames)} frames")
    return frames

def save_video(frames, output_path, fps=10):
    if len(frames) == 0:
        print("❌ No frames to save!")
        return
    
    print(f"🔄 Saving video to: {output_path}")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    
    with tqdm(total=len(frames), desc="Saving frames") as pbar:
        for f in frames:
            out.write(f)
            pbar.update(1)
    
    out.release()
    print(f"✅ Video saved successfully!")

if __name__ == "__main__":
    # Configuration
    frames_dir = "extracted_frames"
    model_path = "models/best4.pt"
    output_path = "output_videos/random_frames_tracking.avi"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("🚀 Starting team tracking analysis on random frames...")
    print("="*60)
    
    # Load extracted frames
    frames = load_extracted_frames(frames_dir)
    
    if not frames:
        print("❌ No frames found! Please run extract_random_frames.py first.")
        exit(1)

    print("\n🔄 Initializing team tracker...")
    tracker = Tracker(model_path)
    
    print("🔄 Running team-based object tracking...")
    tracks = tracker.get_object_tracks(frames, stub_path=None)

    print("\n🔄 Drawing team tracking annotations...")
    annotated = tracker.draw_annotations(frames, tracks)

    print("🔄 Saving team tracking video...")
    save_video(annotated, output_path)
    print(f"✅ Saved team tracking video to: {output_path}")

    # Generate and display tracking quality report
    print("\n" + "="*60)
    quality_report = tracker.get_tracking_quality_report()
    print(quality_report)
    
    # Show sample team tracking data
    print("\n📊 SAMPLE TEAM TRACKING DATA (First 10 frames):")
    print("-" * 50)
    for i in range(min(10, len(tracks['team1_players']))):
        team1_count = len(tracks['team1_players'][i])
        team2_count = len(tracks['team2_players'][i])
        ball_count = len(tracks['ball'][i]) if tracks['ball'][i] else 0
        total_players = team1_count + team2_count
        
        print(f"Frame {i}: Team 1: {team1_count}, Team 2: {team2_count}, Ball: {ball_count}")
        print(f"  Total players: {total_players}")
        
        # Show team balance for first frame
        if i == 0 and total_players > 0:
            team1_ratio = team1_count / total_players * 100
            team2_ratio = team2_count / total_players * 100
            print(f"  Team balance: Team 1 ({team1_ratio:.1f}%) vs Team 2 ({team2_ratio:.1f}%)")

    print("\n🎯 RANDOM FRAMES TRACKING ANALYSIS COMPLETE!")
    print("="*60)
    print("The system has analyzed 100 random frames from the full match.")
    print("Check the output video to see the team-based tracking performance.")
    print("Use the quality report above to identify areas for improvement.")
    print("\nExiting after random frames analysis...") 