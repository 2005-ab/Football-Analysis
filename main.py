from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
import numpy as np
import cv2
import os
import open_clip
import torch
from PIL import Image


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


def extract_clip_embeddings(frames, tracks):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()
    all_embeddings = []
    for f_idx, player_track in enumerate(tracks['players']):
        frame = frames[f_idx]
        for _, info in player_track.items():
            x1, y1, x2, y2 = map(int, info['bbox'])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = preprocess(pil_img).unsqueeze(0)  # Add batch dim
            with torch.no_grad():
                emb = model.encode_image(img_tensor).cpu().numpy().flatten()
            all_embeddings.append(emb)
    all_embeddings = np.array(all_embeddings)
    print(f"Extracted CLIP embeddings shape: {all_embeddings.shape}")
    return all_embeddings


def select_pitch_landmarks(frame):
    print("Please click the 4 pitch corners in the following order: top-left, top-right, bottom-right, bottom-left.")
    points = []
    clone = frame.copy()
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Corners", clone)
    cv2.imshow("Select Corners", clone)
    cv2.setMouseCallback("Select Corners", click_event)
    while len(points) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Select Corners")
    return np.array(points, dtype=np.float32)


def compute_homography(image_points, pitch_size=(105, 68)):
    # Standard football pitch size in meters (FIFA): 105x68
    pitch_points = np.array([
        [0, 0],
        [pitch_size[0], 0],
        [pitch_size[0], pitch_size[1]],
        [0, pitch_size[1]]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(image_points, pitch_points)
    return H


def map_to_pitch(bbox, H):
    # bbox: [x1, y1, x2, y2] -> use center
    xc = (bbox[0] + bbox[2]) / 2
    yc = (bbox[1] + bbox[3]) / 2
    pt = np.array([[xc, yc]], dtype=np.float32)
    pt = np.array([pt])
    pitch_pt = cv2.perspectiveTransform(pt, H)[0][0]
    return pitch_pt


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

    # Extract CLIP embeddings for all player crops
    extract_clip_embeddings(frames, tracks)

    # --- NEW: Pitch mapping ---
    # Hardcoded pitch corners for your video (update these for your video as needed)
    image_points = np.array([
        [100, 50],   # top-left
        [1200, 60],  # top-right
        [1180, 700], # bottom-right
        [90, 690]    # bottom-left
    ], dtype=np.float32)
    H = compute_homography(image_points)
    print("Homography matrix computed.")
    # Map all player and ball positions to pitch coordinates
    for f_idx, (players, ball, team) in enumerate(zip(tracks['players'], tracks['ball'], possession)):
        player_pitch_coords = []
        for _, info in players.items():
            pitch_pt = map_to_pitch(info['bbox'], H)
            player_pitch_coords.append((pitch_pt, info['team']))
        ball_pitch_coord = None
        if ball and 1 in ball:
            ball_pitch_coord = map_to_pitch(ball[1]['bbox'], H)
        print(f"Frame {f_idx}: Team in possession: {team if team else 'Unknown'}")
        print(f"  Player pitch coords: {player_pitch_coords}")
        print(f"  Ball pitch coord: {ball_pitch_coord}")
