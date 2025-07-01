from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
import numpy as np
import cv2
import os


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


def save_video(frames, output_path, fps=24):
    if len(frames) == 0:
        print("❌ No frames to save!")
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def estimate_possession(tracks):
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
        for info in players.values():
            px, py = (info['bbox'][0] + info['bbox'][2]) / 2, (info['bbox'][1] + info['bbox'][3]) / 2
            dist = np.hypot(bx - px, by - py)
            if dist < min_dist:
                min_dist = dist
                possession_team = info.get('team', None)
        possession_per_frame.append(possession_team)
        last_possession = possession_team
    return possession_per_frame


if __name__ == "__main__":
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best4.pt"
    output_path = "output_videos/spurs_clip06_possession.avi"
    cache_path = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = read_video(video_path)

    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, stub_path=cache_path)

    team_assigner = TeamAssigner()
    if len(tracks['players']) > 0 and len(tracks['players'][0]) > 0:
        team_assigner.assign_team_color(frames[0], tracks['players'][0])

    team_colors = {}

    for f_idx, player_track in enumerate(tracks['players']):
        for _, info in player_track.items():
            try:
                col = team_assigner.get_player_color(frames[f_idx], info['bbox'])
                col = tuple(int(x) for x in np.clip(col, 0, 255))
            except:
                col = (0, 0, 255)
            team = team_assigner.get_player_team(frames[f_idx], info['bbox'], 0)
            info['jersey_color'] = col
            info['team'] = team
            if team and team not in team_colors:
                team_colors[team] = col

    # Draw player and ball annotations
    annotated = tracker.draw_annotations(frames, tracks)

    # Estimate possession
    possession = estimate_possession(tracks)

    # Overlay possession info
    for idx, (img, team) in enumerate(zip(annotated, possession)):
        h, w = img.shape[:2]

        # Draw top possession bar
        if team in team_colors:
            cv2.rectangle(img, (0, 0), (w, 30), team_colors[team], -1)
            text_color = (255, 255, 255) if np.mean(team_colors[team]) < 128 else (0, 0, 0)
        else:
            cv2.rectangle(img, (0, 0), (w, 30), (128, 128, 128), -1)
            text_color = (0, 0, 0)

        # Text
        label = f"Possession: Team {team}" if team else "Possession: Unknown"
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)

    save_video(annotated, output_path)
    print(f"✅ Saved to {output_path}")

    for idx, team in enumerate(possession):
        print(f"Frame {idx}: Team in possession: Team {team if team else 'Unknown'}")
