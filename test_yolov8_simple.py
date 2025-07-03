#!/usr/bin/env python3
"""
Simple YOLOv8 test with standard model
"""

from ultralytics import YOLO
import cv2
import os

def test_yolov8_simple():
    print("🔄 Testing YOLOv8 with standard model...")
    
    try:
        # Try with a standard YOLOv8 model first
        model = YOLO('yolov8n.pt')  # Use nano model for testing
        print("✅ YOLOv8 model loaded successfully")
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (128, 128, 128)  # Gray image
        
        print("🔄 Running detection on test image...")
        results = model(test_image, verbose=False)
        print("✅ Detection completed")
        
        print("✅ YOLOv8 is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import numpy as np
    test_yolov8_simple() 