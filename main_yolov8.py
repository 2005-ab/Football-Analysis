#!/usr/bin/env python3
"""
Football Analysis Pipeline using YOLOv8
"""

from trackers.yolov8_tracker import YOLOv8Tracker
from team_assigner.team_assigner import TeamAssigner
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def estimate_possession(tracks):
    """Estimate possession based on ball proximity to players"""
    possession_per_frame = []
    last_possession = None
    
    for f_idx, (players, ball) in enumerate(zip(tracks['players'], tracks['ball'])):
        if not ball or 1 not in ball:
            possession_per_frame.append(last_possession)
            continue
            
        ball_bbox = ball[1]['bbox']
        bx, by = (ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2
        min_dist = float('inf')
        possession_team = None
        
        for track_id, info in players.items():
            if 'team' not in info:
                continue
            px, py = (info['bbox'][0] + info['bbox'][2]) / 2, (info['bbox'][1] + info['bbox'][3]) / 2
            dist = np.hypot(bx - px, by - py)
            if dist < min_dist:
                min_dist = dist
                possession_team = info.get('team', None)
        
        possession_per_frame.append(possession_team)
        last_possession = possession_team
    
    return possession_per_frame


def analyze_tracking_quality(tracks):
    """Analyze tracking quality and consistency"""
    print("\n📊 TRACKING QUALITY ANALYSIS:")
    
    # Track persistence analysis
    all_track_ids = set()
    track_persistence = {}
    
    for frame_idx, players in enumerate(tracks['players']):
        frame_track_ids = set(players.keys())
        all_track_ids.update(frame_track_ids)
        
        for track_id in frame_track_ids:
            if track_id not in track_persistence:
                track_persistence[track_id] = {'start': frame_idx, 'frames': 0}
            track_persistence[track_id]['frames'] += 1
            track_persistence[track_id]['end'] = frame_idx
    
    print(f"Total unique track IDs: {len(all_track_ids)}")
    print(f"Total frames: {len(tracks['players'])}")
    
    # Calculate statistics
    durations = [stats['end'] - stats['start'] + 1 for stats in track_persistence.values()]
    avg_duration = np.mean(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    
    print(f"Average track duration: {avg_duration:.1f} frames")
    print(f"Max track duration: {max_duration} frames")
    print(f"Min track duration: {min_duration} frames")
    
    # Count track switches
    track_switches = 0
    for frame_idx in range(1, len(tracks['players'])):
        prev_ids = set(tracks['players'][frame_idx-1].keys())
        curr_ids = set(tracks['players'][frame_idx].keys())
        new_ids = curr_ids - prev_ids
        track_switches += len(new_ids)
    
    print(f"Total track ID switches: {track_switches}")
    print(f"Average switches per frame: {track_switches / len(tracks['players']):.2f}")


def main():
    # Configuration
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best4.pt"  # Your YOLOv8 model
    output_path = "output_videos/spurs_clip06_yolov8.avi"
    cache_path = "stubs/yolov8_tracks.pkl"
    
    # For testing, limit frames
    MAX_FRAMES = 100  # Set to None for full video processing
    
    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    print("🚀 Starting YOLOv8 Football Analysis Pipeline...")
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load video
    frames = read_video(video_path, max_frames=MAX_FRAMES)
    
    # Initialize YOLOv8 tracker
    print("🔄 Initializing YOLOv8 tracker...")
    tracker = YOLOv8Tracker(
        model_path=model_path,
        conf_threshold=0.3,  # Adjust confidence threshold
        iou_threshold=0.5    # Adjust IoU threshold
    )
    
    # Run detection and tracking
    print("🔄 Running YOLOv8 detection and tracking...")
    tracks = tracker.detect_and_track(frames, stub_path=cache_path)
    
    # Analyze tracking quality
    analyze_tracking_quality(tracks)
    
    # Initialize team assigner
    print("🔄 Initializing team assigner...")
    team_assigner = TeamAssigner()
    
    # Assign teams to players
    if len(tracks['players']) > 0 and len(tracks['players'][0]) > 0:
        team_assigner.assign_team_color(frames[0], tracks['players'][0])
    
    team_colors = {}
    
    print("🔄 Assigning teams and colors...")
    with tqdm(total=len(tracks['players']), desc="Processing frames") as pbar:
        for f_idx, player_track in enumerate(tracks['players']):
            for track_id, info in player_track.items():
                try:
                    col = team_assigner.get_player_color(frames[f_idx], info['bbox'])
                    col = tuple(int(x) for x in np.clip(col, 0, 255))
                except:
                    col = (0, 0, 255)
                
                team = team_assigner.get_player_team(frames[f_idx], info['bbox'], track_id)
                info['jersey_color'] = col
                info['team'] = team
                
                if team and team not in team_colors:
                    team_colors[team] = col
            pbar.update(1)
    
    # Draw annotations
    print("🔄 Drawing annotations...")
    annotated = tracker.draw_annotations(frames, tracks)
    
    # Estimate possession
    print("🔄 Estimating possession...")
    possession = estimate_possession(tracks)
    
    # Overlay possession info
    print("🔄 Adding overlays...")
    with tqdm(total=len(annotated), desc="Adding overlays") as pbar:
        for idx, (img, team) in enumerate(zip(annotated, possession)):
            h, w = img.shape[:2]
            
            # Draw possession bar
            if team in team_colors:
                cv2.rectangle(img, (0, 0), (w, 30), team_colors[team], -1)
                text_color = (255, 255, 255) if np.mean(team_colors[team]) < 128 else (0, 0, 0)
            else:
                cv2.rectangle(img, (0, 0), (w, 30), (128, 128, 128), -1)
                text_color = (0, 0, 0)
            
            # Add text overlays
            label = f"Possession: Team {team}" if team else "Possession: Unknown"
            cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
            
            # Add tracking statistics
            if idx < len(tracks['players']):
                num_players = len(tracks['players'][idx])
                cv2.putText(img, f"Players: {num_players}", (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
                
                # Add frame number
                cv2.putText(img, f"Frame: {idx}", (30, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
            pbar.update(1)
    
    # Save video
    save_video(annotated, output_path)
    
    # Print summary
    print("\n📊 POSSESSION SUMMARY (first 10 frames):")
    for idx, team in enumerate(possession[:10]):
        print(f"Frame {idx}: Team in possession: Team {team if team else 'Unknown'}")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📹 Output video: {output_path}")
    print(f"📊 Tracking cache: {cache_path}")


if __name__ == "__main__":
    main() 