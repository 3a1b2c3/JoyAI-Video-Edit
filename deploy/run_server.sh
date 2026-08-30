#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JOYOMNI_CONDA_SH="${JOYOMNI_CONDA_SH:-}"
JOYOMNI_CONDA_ENV="${JOYOMNI_CONDA_ENV:-}"
if [ -n "$JOYOMNI_CONDA_ENV" ]; then
  : "${NVCC_PREPEND_FLAGS:=}" "${NVCC_APPEND_FLAGS:=}"
  export NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
  if [ -n "$JOYOMNI_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$JOYOMNI_CONDA_SH"
  fi
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
  conda activate "$JOYOMNI_CONDA_ENV"
  hash -r
fi

cd "$HERE"

JOYOMNI_CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$HERE/deps/cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$JOYOMNI_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$JOYOMNI_CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$JOYOMNI_CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
# joyomni_ops/ (a nested repo checkout: $HERE/joyomni_ops/joyomni_ops/__init__.py)
# has no __init__.py at $HERE/joyomni_ops itself, so with only $HERE on the
# path, Python can resolve "import joyomni_ops" to that outer directory as an
# empty PEP 420 namespace package instead of the real one -- symptom is
# ImportError for real joyomni_ops functions, "(unknown location)" in the
# error (namespace packages have no single __file__). Putting the real
# package's containing directory first makes Python find the actual
# (regular, __init__.py-having) package immediately, which always wins over
# a namespace candidate.
export PYTHONPATH="$HERE/joyomni_ops:$HERE"
echo "PYTHONPATH=$PYTHONPATH"

export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"
export JOYOMNI_CUDA_GRAPH="${JOYOMNI_CUDA_GRAPH:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-0}"
export JOYOMNI_FP8_FAST_ACCUM="${JOYOMNI_FP8_FAST_ACCUM:-0}"
export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-0}"

# Prompt enhancement (empty -> disabled, the raw user prompt is used as-is)
export PE_MODEL="${PE_MODEL:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

RECORD_DIR="${JOYOMNI_RECORD_DIR:-$HERE/recordings}"

CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$HERE/deps/checkpoints}"
DIT_CKPT="${JOYOMNI_DIT_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth}"
VAE_CKPT="${JOYOMNI_VAE_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/vae}"
TE_CKPT="${JOYOMNI_TEXT_ENCODER_CKPT:-$CKPT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${JOYOMNI_FACE_ONNX:-$CKPT_ROOT/face_detection_yunet_2023mar.onnx}"
PERSON_ONNX="${JOYOMNI_PERSON_ONNX:-$CKPT_ROOT/yolov8n.onnx}"

DEVICE="${JOYOMNI_DEVICE:-cuda:0}"
HOST="${JOYOMNI_HOST:-0.0.0.0}"
PORT="${JOYOMNI_PORT:-8080}"

python xvideo/serving/serve_joyomni_streaming.py \
  --dit-ckpt          "$DIT_CKPT" \
  --vae-ckpt          "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx   "$FACE_ONNX" \
  --person-detector-onnx "$PERSON_ONNX" \
  --record-dir "$RECORD_DIR" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --width "${JOYOMNI_WIDTH:-840}" --height "${JOYOMNI_HEIGHT:-480}" \
  --fps "${JOYOMNI_FPS:-24}" \
  --host "$HOST" --port "$PORT" \
  "$@"
