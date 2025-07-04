from trackers.tracker import Tracker
import numpy as np
import cv2
import os
from tqdm import tqdm


def read_video(video_path, max_frames=None):
    """Read video with optional frame limit for testing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    print(f"🔄 Loading video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"📹 Processing first {max_frames} frames for testing")
    
    with tqdm(total=total_frames, desc="Loading frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            frames.append(frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"✅ Loaded {len(frames)} frames")
    return frames


def save_video(frames, output_path, fps=24):
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
    video_path = "clips/spurs_build_up/spurs_clip04.mp4"
    model_path = "models/best4.pt"
    output_path = "output_videos/team_tracking_ellipse.avi"
    cache_path = None
    
    # For testing, limit frames to first 100
    MAX_FRAMES = 100  # Set to None for full video processing

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("🚀 Starting team-based tracking analysis...")
    print("="*60)
    
    # Load video with frame limit for testing
    frames = read_video(video_path, max_frames=MAX_FRAMES)

    print("\n🔄 Initializing team tracker...")
    tracker = Tracker(model_path)
    
    print("🔄 Running object tracking...")
    tracks = tracker.get_object_tracks(frames, stub_path=cache_path)

    # Use the draw_annotations method from Tracker
    print("🔄 Drawing tracking annotations...")
    annotated = tracker.draw_annotations(frames, tracks)

    print("🔄 Saving tracking video...")
    save_video(annotated, output_path)
    print(f"✅ Saved to {output_path}")

    # Generate and display tracking quality report
    print("\n" + "="*60)
    quality_report = tracker.get_tracking_quality_report()
    print(quality_report)
    
    # Show sample team tracking data
    print("\n📊 SAMPLE TEAM TRACKING DATA (First 5 frames):")
    print("-" * 50)
    for i in range(min(5, len(tracks['team1_players']))):
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

    print("\n🎯 TEAM TRACKING ANALYSIS COMPLETE!")
    print("="*60)
    print("The system now tracks players by team (Team 1 vs Team 2) instead of individual IDs.")
    print("Players are assigned to teams based on jersey color and position.")
    print("Check the output video to see the team-based tracking performance.")
    print("Use the quality report above to identify areas for improvement.")
    print("\nExiting after team tracking analysis...")
