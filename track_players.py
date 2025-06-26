from ultralytics import YOLO
import cv2
import os
import pandas as pd

input_dir = "data/tiki_taka"
output_dir = "data/tiki_taka_tracks_clean"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8s.pt")

for i, filename in enumerate(sorted(os.listdir(input_dir)), 1):
    if not filename.endswith((".mp4", ".avi")):
        continue

    video_path = os.path.join(input_dir, filename)
    clip_name = f"clip{i:02d}"
    print(f"▶️ Processing: {clip_name}")

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        result = results[0]

        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
            boxes = result.boxes.xywh.cpu().numpy()  # [x_center, y_center, w, h]

            for obj_id, box in zip(ids, boxes):
                all_detections.append([
                    frame_num,
                    int(obj_id),
                    box[0], box[1], box[2], box[3]
                ])
        
        frame_num += 1

    cap.release()

    df = pd.DataFrame(all_detections, columns=["frame", "id", "x_center", "y_center", "width", "height"])
    df.to_csv(os.path.join(output_dir, f"{clip_name}_tracks.csv"), index=False)
    print(f"✅ Saved: {clip_name}_tracks.csv")
