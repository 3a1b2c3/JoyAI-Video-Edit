#!/bin/bash
# Precompile Triton kernels (1 minute, one-time cost)

set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="$PWD/deploy:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"

mkdir -p "$TORCH_HOME" "$HF_HOME"

echo "Precompiling Triton kernels (warmup)..."
echo ""

python - << 'EOF'
import sys
import torch
sys.path.insert(0, 'deploy')

from xvideo.serving.joyomni_streaming import JoyOmniRuntime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[1/2] Loading models...")
runtime = JoyOmniRuntime.load(
    dit_ckpt="deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth",
    vae_ckpt="deploy/deps/checkpoints/JoyAI-Video-Edit/vae",
    text_encoder_ckpt="deploy/deps/checkpoints/MiMo-VL-7B-RL-2508",
    device=device,
)
print("✓ Models loaded")
print("")

print("[2/2] Warming up (1 dummy step)...")
# Just verify inference works with 1 step
result = runtime.infer(
    prompt="test",
    num_frames=1,
    height=480,
    width=864,
    num_steps=1,  # 1 step = fast warmup
    seed=42,
)
print("✓ Warmup complete")
print("")

print("=" * 70)
print("✅ Precompilation done! Inference is now fast.")
print("=" * 70)
EOF

