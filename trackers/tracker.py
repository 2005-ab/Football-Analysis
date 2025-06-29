import os
from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np  # 🔺 Needed for triangle drawing

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, conf=0.2, batch_size=16):
        dets = []
        for i in range(0, len(frames), batch_size):
            dets_batch = self.model.predict(frames[i:i+batch_size], conf=conf)
            dets.extend(dets_batch)
        return dets

    def get_object_tracks(self, frames, stub_path=None):
        if stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path, 'rb'))

        dets = self.detect_frames(frames)
        tracks = {"players": [], "ball": [], "referees": []}

        for idx, det in enumerate(dets):
            names = det.names
            inv = {v: k for k, v in names.items()}
            sup = sv.Detections.from_ultralytics(det)

            # map goalkeepers → players
            for i, cid in enumerate(sup.class_id):
                if names[cid] == "goalkeeper":
                    sup.class_id[i] = inv["player"]

            with_tracks = self.tracker.update_with_detections(sup)
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for b in with_tracks:
                bbox = b[0].tolist()
                cid = b[3]
                tid = b[4]
                cls = names[cid]
                if cls == "player":
                    tracks["players"][idx][tid] = {"bbox": bbox}
                elif cls == "referee":
                    tracks["referees"][idx][tid] = {"bbox": bbox}

            # ball (id=1)
            for b in sup:
                bbox = b[0].tolist()
                cid = b[3]
                if names[cid] == "ball":
                    tracks["ball"][idx][1] = {"bbox": bbox}

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

    # 🔺 NEW: Ball triangle drawing
    def draw_triangle(self, frame, bbox, color=(0, 255, 255)):
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yt = int(y1)
        w = int((x2 - x1) * 0.6)
        h = int((x2 - x1) * 0.6)
        points = np.array([
            [xc, yt + h],         # bottom point (triangle tip)
            [xc - w // 2, yt],    # top-left
            [xc + w // 2, yt]     # top-right
        ])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        return frame

    def draw_annotations(self, frames, tracks):
        out = []
        for i, frame in enumerate(frames):
            img = frame.copy()

            # Draw players with ellipse and ID
            for tid, info in tracks["players"][i].items():
                img = self.draw_ellipse(img, info["bbox"], track_id=tid)

            # 🔺 Draw the ball with triangle
            for _, info in tracks["ball"][i].items():
                img = self.draw_triangle(img, info["bbox"], color=(0, 255, 255))  # yellow

            out.append(img)
        return out
