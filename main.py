from ultralytics import YOLO
from utils.video_utils import save_video, read_video

model = YOLO('models/best3.pt')

results=model.predict('clips\spurs_build_up\spurs_clip03.mp4',save=True)
print(results[0])

for box in results[0].boxes:
    print(box)
    