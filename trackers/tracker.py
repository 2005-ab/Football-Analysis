import os
import cv2
import pickle
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.detection.core import Detections


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, conf=0.2, batch_size=16):
        dets = []
        for i in range(0, len(frames), batch_size):
            dets_batch = self.model.predict(frames[i:i + batch_size], conf=conf)
            dets.extend(dets_batch)
        return dets

    def get_object_tracks(self, frames, stub_path=None):
        if stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path, 'rb'))

        dets = self.detect_frames(frames)
        tracks = {"players": [], "ball": []}

        prev_player_tracks = {}
        prev_ball_track = {}

        for idx, det in enumerate(dets):
            boxes = det.boxes
            names = det.names
            inv = {v: k for k, v in names.items()}

            if boxes is None or len(boxes) == 0:
                tracks["players"].append(prev_player_tracks.copy())
                tracks["ball"].append(prev_ball_track.copy())
                continue

            valid_indices = []
            for i, box in enumerate(boxes):
                cls_id = box.cls
                cls_id_int = inv.get(cls_id, -1) if isinstance(cls_id, str) else int(cls_id)
                cls_name = names[cls_id_int]
                if cls_name in ["player", "goalkeeper", "ball"]:
                    valid_indices.append(i)

            if not valid_indices:
                tracks["players"].append(prev_player_tracks.copy())
                tracks["ball"].append(prev_ball_track.copy())
                continue

            boxes = boxes[valid_indices]

            # Convert goalkeeper → player
            cls_tensor = boxes.cls
            for i in range(len(cls_tensor)):
                cls_id = cls_tensor[i]
                cls_id_val = int(cls_id.item()) if isinstance(cls_id, torch.Tensor) else int(cls_id)
                if names[cls_id_val] == "goalkeeper":
                    cls_tensor[i] = inv["player"]

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            cls_fixed = [inv.get(c, -1) if isinstance(c, str) else int(c) for c in cls]
            cls = np.array(cls_fixed)

            sup = Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            with_tracks = self.tracker.update_with_detections(sup)

            tracks["players"].append({})
            tracks["ball"].append({})
            current_players = {}
            ball_found = False

            for det in with_tracks:
                bbox = det[0].tolist()
                class_id = det[3]
                track_id = int(det[4])
                class_id_int = inv.get(class_id, -1) if isinstance(class_id, str) else int(class_id)
                cls = names[class_id_int]
                if cls in ["player", "goalkeeper"]:
                    current_players[track_id] = {"bbox": bbox}
                elif cls == "ball":
                    tracks["ball"][idx][1] = {"bbox": bbox}
                    ball_found = True

            if not current_players and prev_player_tracks:
                current_players = prev_player_tracks.copy()
            if not ball_found and 1 in prev_ball_track:
                tracks["ball"][idx][1] = prev_ball_track[1]

            tracks["players"][idx] = current_players
            prev_player_tracks = current_players.copy()
            if 1 in tracks["ball"][idx]:
                prev_ball_track = {1: tracks["ball"][idx][1]}

        if stub_path:
            pickle.dump(tracks, open(stub_path, 'wb'))
        return tracks

    def draw_ellipse(self, frame, bbox, color=(255, 255, 255), thickness=4):
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yb = int(y2)
        W = int((x2 - x1) * 0.8)
        H = int(W * 0.5)
        yc = yb - H // 2 - 10
        cv2.ellipse(frame, (xc, yc), (W, H), 0, -30, 235, color, thickness, cv2.LINE_AA)
        return frame

    def draw_triangle(self, frame, bbox, color=(0, 255, 255)):
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yt = int(y1)
        w = int((x2 - x1) * 0.6)
        h = int((x2 - x1) * 0.6)
        points = np.array([
            [xc, yt + h],
            [xc - w // 2, yt],
            [xc + w // 2, yt]
        ])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        return frame

    def draw_annotations(self, frames, tracks):
        out = []
        for i, frame in enumerate(frames):
            img = frame.copy()

            for _, info in tracks["players"][i].items():
                color = tuple(int(x) for x in info.get("jersey_color", (0, 0, 255)))
                img = self.draw_ellipse(img, info["bbox"], color=color)

            for _, info in tracks["ball"][i].items():
                img = self.draw_triangle(img, info["bbox"], color=(0, 255, 255))

            out.append(img)
        return out
