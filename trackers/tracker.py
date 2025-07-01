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

        for idx, det in enumerate(dets):
            boxes = det.boxes
            names = det.names
            inv = {v: k for k, v in names.items()}

            if boxes is None or len(boxes) == 0:
                # No detections in this frame
                tracks["players"].append({})
                tracks["ball"].append({})
                continue

            # Filter out referees and keep only players and ball
            valid_indices = []
            for i, box in enumerate(boxes):
                cls_name = names[int(box.cls)]
                if cls_name in ["player", "goalkeeper", "ball"]:
                    valid_indices.append(i)
            
            if not valid_indices:
                tracks["players"].append({})
                tracks["ball"].append({})
                continue

            # Keep only valid detections
            boxes = boxes[valid_indices]

            # Fix goalkeeper class IDs → player
            for box in boxes:
                if names[int(box.cls)] == "goalkeeper":
                    box.cls = torch.tensor(inv["player"]).to(box.cls.device)

            # Parse into supervision format
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            sup = Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=cls
            )

            # Track objects
            with_tracks = self.tracker.update_with_detections(sup)

            # Initialize frame
            tracks["players"].append({})
            tracks["ball"].append({})

            for det in with_tracks:
                bbox = det[0].tolist()
                class_id = int(det[3])
                track_id = int(det[4])
                cls = names[class_id]

                if cls == "player":
                    tracks["players"][idx][track_id] = {"bbox": bbox}

            # Ball → Not tracked, only detect
            for box, class_id in zip(xyxy, cls):
                if names[class_id] == "ball":
                    tracks["ball"][idx][1] = {"bbox": box.tolist()}

        if stub_path:
            pickle.dump(tracks, open(stub_path, 'wb'))
        return tracks

    def draw_ellipse(self, frame, bbox, track_id=None, color=(255, 255, 255), thickness=4):
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yb = int(y2)
        W = int((x2 - x1) * 0.8)
        H = int(W * 0.5)
        yc = yb - H // 2 - 10

        cv2.ellipse(frame, (xc, yc), (W, H), 0, -30, 235, color, thickness, cv2.LINE_AA)
        if track_id is not None:
            rw, rh = 40, 28
            rx1 = xc - rw // 2
            ry1 = yc + H // 2 + 6
            cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), color, cv2.FILLED)
            scale = 0.8 if track_id < 100 else 0.6
            tx = rx1 + (8 if track_id < 100 else 4)
            cv2.putText(frame, str(track_id), (tx, ry1 + rh - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
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

            for tid, info in tracks["players"][i].items():
                color = tuple(int(x) for x in info.get("jersey_color", (0, 0, 255)))
                img = self.draw_ellipse(img, info["bbox"], track_id=tid, color=color)

            for _, info in tracks["ball"][i].items():
                img = self.draw_triangle(img, info["bbox"], color=(0, 255, 255))

            out.append(img)
        return out 