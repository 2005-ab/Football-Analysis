from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # ✅ Fix class label mismatches here (no retraining needed)
            class_fix_map = {
                'ball': 'player',
                'player': 'ball',
                'goalkeeper': 'referee',
                'referee': 'goalkeeper'
            }

            for i, class_id in enumerate(detection_supervision.class_id):
                original_name = cls_names[class_id]
                if original_name in class_fix_map:
                    corrected_name = class_fix_map[original_name]
                    detection_supervision.class_id[i] = cls_names_inv[corrected_name]

            print(f"Frame {frame_num}:")
            print(detection_supervision)
