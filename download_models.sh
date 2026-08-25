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
echo "Done: DiT, VAE, MiMo-VL, YuNet face detector."
echo
echo "NOT downloaded (needs a separate throwaway conda env, see DEPLOYMENT.md 3c):"
echo "  yolov8n.onnx (person detector) -- must be exported locally at imgsz=320;"
echo "  pre-exported copies on the Hub won't load (wrong input size)."
echo "  conda create -n yolo-export python=3.10 -y && conda activate yolo-export"
echo "  pip install --index-url https://pypi.org/simple/ \\"
echo "    --extra-index-url https://download.pytorch.org/whl/cpu ultralytics onnx onnxslim"
echo "  pip uninstall -y opencv-python && pip install --index-url https://pypi.org/simple/ opencv-python-headless"
echo "  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)\""
echo "  conda deactivate && mv yolov8n.onnx '$CKPT_ROOT/' && conda env remove -n yolo-export -y"
