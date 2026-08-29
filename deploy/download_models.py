#!/usr/bin/env python3
"""Download YOLOv8n ONNX model using Python HF API."""
import os
import sys
from pathlib import Path

def download_yolo_model():
    """Download YOLOv8n ONNX to checkpoints."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install --break-system-packages ultralytics")
        from ultralytics import YOLO

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "deps", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    yolo_path = os.path.join(ckpt_dir, "yolov8n.onnx")
    print(f"Downloading YOLOv8n ONNX...")
    print(f"Cache: {ckpt_dir}")

    try:
        model = YOLO("yolov8n.pt")
        model.export(format="onnx", imgsz=640)
        print("✓ Download complete!")
        return 0
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(download_yolo_model())
