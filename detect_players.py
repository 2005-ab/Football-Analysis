from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8x.pt")  # pretrained COCO model (person class = 0)

input_folder = "clips\city_build_up"
output_folder = "clips\city_build_up_annotated"
os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):
    if not filename.endswith(".mp4"):
        continue

    cap = cv2.VideoCapture(os.path.join(input_folder, filename))
    out_path = os.path.join(output_folder, filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved annotated video: {out_path}")
