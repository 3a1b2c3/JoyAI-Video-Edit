#!/bin/bash
# Downloads all JoyAI-Video-Edit checkpoints into deploy/deps/checkpoints/
# (~51 GB total), per DEPLOYMENT.md section 3. Run from the repo root.
#
#   bash download_models.sh
#
# Skips the optional YOLOv8n export (needs a throwaway conda env + a specific
# imgsz=320/opset=12 export -- see DEPLOYMENT.md 3c for that by hand). Without
# it the server just disables the person-presence gate (edits run unconditionally).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-deploy/deps/checkpoints}"
mkdir -p "$CKPT_ROOT"

command -v hf >/dev/null 2>&1 || { echo "ERROR: 'hf' CLI not found (pip install -U huggingface_hub)"; exit 1; }

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
echo "=== [3/3] YuNet face detector (optional -- face-presence gate) ==="
curl -L -o "$CKPT_ROOT/face_detection_yunet_2023mar.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

echo
echo "=== [4/4] YOLOv8n person detector (optional -- person-presence gate) ==="
if [ -f "$CKPT_ROOT/yolov8n.onnx" ]; then
  echo "✓ yolov8n.onnx already exists"
else
  echo "Downloading YOLOv8n v8.1.0 (OpenCV-compatible)..."
  curl -L -o "$CKPT_ROOT/yolov8n.onnx" \
    https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx
  if [ -f "$CKPT_ROOT/yolov8n.onnx" ]; then
    echo "✓ Downloaded yolov8n.onnx"
  else
    echo "⚠️  yolov8n.onnx download failed (optional, server will disable person-presence gate)"
  fi
fi

echo
echo "=== Done ==="
echo "  $CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"
echo "  $CKPT_ROOT/JoyAI-Video-Edit/vae/{config.json, diffusion_pytorch_model.safetensors}"
echo "  $CKPT_ROOT/MiMo-VL-7B-RL-2508/"
echo "  $CKPT_ROOT/face_detection_yunet_2023mar.onnx"
echo "  $CKPT_ROOT/yolov8n.onnx"
