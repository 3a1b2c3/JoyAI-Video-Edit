#!/bin/bash
# Download YOLOv8n ONNX model for person detection gate
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/deploy/deps/checkpoints"
mkdir -p "$CKPT_DIR"

YOLO_PATH="$CKPT_DIR/yolov8n.onnx"

if [ -f "$YOLO_PATH" ]; then
  echo "✓ YOLOv8n already exists at $YOLO_PATH"
  exit 0
fi

echo "⏳ Exporting YOLOv8n to ONNX using Ultralytics library..."
python3 << 'PYTHON_EOF'
import os
import sys

ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy/deps/checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

yolo_path = os.path.join(ckpt_dir, "yolov8n.onnx")

try:
    from ultralytics import YOLO
    print(f"Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print(f"Exporting to ONNX: {yolo_path}")
    model.export(format="onnx", imgsz=640)
    print(f"✓ Exported to {yolo_path}")
except ImportError:
    print("❌ ultralytics library not found. Install with: pip install ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"❌ Export failed: {e}")
    sys.exit(1)
PYTHON_EOF
