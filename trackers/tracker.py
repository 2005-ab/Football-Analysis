# tracker.py
import torch
import cv2
import os
from my_utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        # ✅ Force reload to avoid 'grid' errors from old cached models
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    def detect(self, frame):
        results = self.model(frame)
        return results

    def draw_boxes(self, frame, results):
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cords[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cls = int(labels[i])
                label = self.model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return frame

    def process_video(self, frames):
        processed_frames = []
        for frame in frames:
            results = self.detect(frame)
            annotated = self.draw_boxes(frame, results)
            processed_frames.append(annotated)
        return processed_frames
