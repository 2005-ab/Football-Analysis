#!/usr/bin/env python3
"""
Test script to validate tracking improvements
"""

from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
import cv2
import os
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def analyze_tracking_consistency(tracks):
    """Analyze tracking consistency across frames"""
    print("\n=== TRACKING CONSISTENCY ANALYSIS ===")
    
    # Track ID persistence
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
    
    # Show track persistence statistics
    print("\nTrack Persistence:")
    for track_id, stats in sorted(track_persistence.items()):
        duration = stats['end'] - stats['start'] + 1
        print(f"  Track {track_id}: Frames {stats['start']}-{stats['end']} ({duration} frames)")
    
    # Calculate average track duration
    durations = [stats['end'] - stats['start'] + 1 for stats in track_persistence.values()]
    avg_duration = np.mean(durations)
    print(f"\nAverage track duration: {avg_duration:.1f} frames")
    
    # Count track switches (when a player gets a new ID)
    track_switches = 0
    for frame_idx in range(1, len(tracks['players'])):
        prev_ids = set(tracks['players'][frame_idx-1].keys())
        curr_ids = set(tracks['players'][frame_idx].keys())
        
        # Count new IDs that appear
        new_ids = curr_ids - prev_ids
        track_switches += len(new_ids)
    
    print(f"Total track ID switches: {track_switches}")
    print(f"Average switches per frame: {track_switches / len(tracks['players']):.2f}")

def test_team_assignment_consistency(tracks, team_assigner, frames):
    """Test team assignment consistency"""
    print("\n=== TEAM ASSIGNMENT CONSISTENCY ===")
    
    team_changes = 0
    total_assignments = 0
    
    for frame_idx, players in enumerate(tracks['players']):
        for track_id, info in players.items():
            total_assignments += 1
            team = team_assigner.get_player_team(frames[frame_idx], info['bbox'], track_id)
            confidence = team_assigner.get_team_confidence(track_id)
            
            if frame_idx > 0 and track_id in tracks['players'][frame_idx-1]:
                prev_team = team_assigner.get_player_team(frames[frame_idx-1], 
                                                        tracks['players'][frame_idx-1][track_id]['bbox'], 
                                                        track_id)
                if prev_team != team:
                    team_changes += 1
                    print(f"  Frame {frame_idx}: Track {track_id} team changed from {prev_team} to {team} (conf: {confidence:.2f})")
    
    print(f"Total team assignments: {total_assignments}")
    print(f"Team changes: {team_changes}")
    print(f"Team change rate: {team_changes / total_assignments * 100:.1f}%")

def main():
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best4.pt"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("🔄 Loading video...")
    frames = read_video(video_path)
    print(f"✅ Loaded {len(frames)} frames")
    
    print("🔄 Initializing tracker...")
    tracker = Tracker(model_path)
    
    print("🔄 Running tracking...")
    tracks = tracker.get_object_tracks(frames)
    
    print("🔄 Initializing team assigner...")
    team_assigner = TeamAssigner()
    if len(tracks['players']) > 0 and len(tracks['players'][0]) > 0:
        team_assigner.assign_team_color(frames[0], tracks['players'][0])
    
    # Analyze tracking consistency
    analyze_tracking_consistency(tracks)
    
    # Test team assignment consistency
    test_team_assignment_consistency(tracks, team_assigner, frames)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 