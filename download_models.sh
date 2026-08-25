#!/bin/bash
# Download all checkpoints per DEPLOYMENT.md section 3 (~51 GB total).
# Resumable -- hf/curl both support resuming partial downloads, so re-running this
# after an interruption picks up where it left off rather than restarting.
#
#   JOYOMNI_CKPT_ROOT=~/joyai_checkpoints bash download_models.sh   # e.g. to avoid a full C: drive
#   bash download_models.sh                                        # defaults to deploy/deps/checkpoints
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$HERE/deploy/deps/checkpoints}"
mkdir -p "$CKPT_ROOT"
echo "Checkpoints will land under: $CKPT_ROOT"
echo

echo "=== [1/3] DiT + xVAE (jdopensource/JoyAI-Video-Edit) ==="
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir "$CKPT_ROOT/JoyAI-Video-Edit" \
  --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"
echo

echo "=== [2/3] Text/vision encoder (XiaomiMiMo/MiMo-VL-7B-RL-2508) ==="
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir "$CKPT_ROOT/MiMo-VL-7B-RL-2508"
echo

echo "=== [3/3] YuNet face detector (optional) ==="
curl -L -o "$CKPT_ROOT/face_detection_yunet_2023mar.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

echo
echo "=== [4/4] YOLOv8n person detector (exporting locally) ==="
echo

# Check if already exists
if [ -f "$CKPT_ROOT/yolov8n.onnx" ]; then
    echo "✓ YOLOv8n already exists"
else
    # Try to export with current environment
    python << 'PYEOF'
import os
from pathlib import Path

ckpt_root = Path(os.environ['CKPT_ROOT'])
yolo_file = ckpt_root / 'yolov8n.onnx'

if yolo_file.exists():
    print(f"✓ YOLOv8n exists ({yolo_file.stat().st_size / (1024**2):.1f} MB)")
else:
    try:
        print("Exporting YOLOv8n to ONNX (imgsz=320)...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.export(format='onnx', imgsz=320, opset=12)

        # Move to checkpoints
        src = Path('yolov8n.onnx')
        if src.exists():
            src.rename(yolo_file)
            print(f"✓ YOLOv8n exported ({yolo_file.stat().st_size / (1024**2):.1f} MB)")
        else:
            print("✗ Export failed")
            exit(1)
    except ImportError:
        print("⚠ ultralytics not installed")
        print("  Install with: pip install ultralytics")
        print("  Then re-run: bash download_models.sh")
        exit(1)
PYEOF
fi

echo
echo "=== All Models Downloaded ==="
echo
echo "Checkpoints:"
du -sh "$CKPT_ROOT"/* 2>/dev/null | sed 's/^/  /'
echo
echo "Total: $(du -sh "$CKPT_ROOT" | cut -f1)"
echo
echo "Ready to run: bash run_server_best.sh"
