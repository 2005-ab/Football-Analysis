#!/usr/bin/env python3
"""
Simple test script for YOLOv8 detection
"""

from ultralytics import YOLO
import cv2
import os

def test_yolov8():
    # Test with a simple image or first frame of video
    video_path = "clips/spurs_build_up/spurs_clip06.mp4"
    model_path = "models/best4.pt"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    print("🔄 Loading YOLOv8 model...")
    try:
        model = YOLO(model_path)
        print("✅ YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("🔄 Loading video frame...")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video frame")
        return
    
    print(f"✅ Loaded frame with shape: {frame.shape}")
    
    print("🔄 Running YOLOv8 detection...")
    try:
        results = model(frame, conf=0.3, verbose=False)
        print("✅ YOLOv8 detection completed")
        
        # Print detection results
        for result in results:
            if result.boxes is not None:
                print(f"Found {len(result.boxes)} detections")
                for i in range(min(5, len(result.boxes))):  # Show first 5
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    confidence = result.boxes.conf[i].cpu().numpy()
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    class_name = result.names[class_id]
                    print(f"  {class_name}: conf={confidence:.2f}, bbox={bbox}")
            else:
                print("No detections found")
                
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return
    
    print("✅ YOLOv8 test completed successfully!")

if __name__ == "__main__":
    test_yolov8() 