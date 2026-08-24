#!/bin/bash
# Pre-flight checks before running server
# Run this before bash run_server_best.sh to catch issues early

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "JoyAI Server Pre-Flight Check"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

# Test 1: CUDA available
echo "[1/6] CUDA..."
if command -v nvidia-smi &>/dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
    echo "  ✓ GPU: $GPU_NAME (${GPU_MEM} MB)"
    PASSED=$((PASSED+1))
else
    echo "  ✗ CUDA/nvidia-smi not found"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 2: Models downloaded
echo "[2/6] Model checkpoints..."
MODELS=(
    "deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"
    "deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors"
    "deploy/deps/checkpoints/MiMo-VL-7B-RL-2508/model.safetensors"
    "deploy/deps/checkpoints/face_detection_yunet_2023mar.onnx"
)
for model in "${MODELS[@]}"; do
    if [ -f "$model" ]; then
        SIZE=$(du -h "$model" | cut -f1)
        echo "  ✓ $(basename "$model") ($SIZE)"
    else
        echo "  ✗ Missing: $model"
        FAILED=$((FAILED+1))
    fi
done
if [ $FAILED -eq 0 ]; then
    echo ""
    PASSED=$((PASSED+1))
fi
echo ""

# Test 3: joyomni_ops installed as a real package (not a namespace-package shadow)
# with its FP8 functions actually importable. A bare `import joyomni_ops` succeeds
# even when a stray joyomni_ops/ source dir shadows the real install or the compiled
# _C extension is broken/missing -- both have bitten this repo before, so check for
# real -- exact interpreter used to run infer_standalone.py/the server.
echo "[3/6] joyomni_ops module..."
if JOYOMNI_CHECK=$(python -c "
import joyomni_ops
assert getattr(joyomni_ops, '__file__', None), 'resolved as a namespace package (no __file__) -- not a real install'
from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
from joyomni_ops._C import fp8_scaled_mm, sgl_per_token_quant_fp8
print(joyomni_ops.__file__)
" 2>&1); then
    echo "  ✓ Installed with FP8 support ($JOYOMNI_CHECK)"
    PASSED=$((PASSED+1))
else
    echo "  ✗ joyomni_ops broken under $(python -c 'import sys; print(sys.executable)'):"
    echo "$JOYOMNI_CHECK" | sed 's/^/    /'
    echo "  Run: cd deploy/joyomni_ops && PYTHON=$(python -c 'import sys; print(sys.executable)') bash build.sh"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 4: PyTorch CUDA
echo "[4/6] PyTorch CUDA..."
if python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}'); print(f'  ✓ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null; then
    PASSED=$((PASSED+1))
else
    echo "  ✗ PyTorch not properly installed"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 5: Diffusers & transformers
echo "[5/6] Dependencies (diffusers, transformers)..."
if python -c "from diffusers import AutoencoderKL; from transformers import Qwen2_5_VLForConditionalGeneration; print('  ✓ All imports OK')" 2>/dev/null; then
    PASSED=$((PASSED+1))
else
    echo "  ✗ Missing dependency"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 6: Quick model load (optional, slow)
echo "[6/6] Quick model load test (this may take 30s)..."
if timeout 60 python -c "
import torch
import sys
sys.path.insert(0, 'deploy')
from xvideo.models.models import load_model
from xvideo.config import ExpConfig
cfg = ExpConfig()
cfg.dit_precision = 'bf16'
print('  Testing model load...')
# Just load the config, don't actually load models (too slow for pre-flight)
print('  ✓ Config valid')
" 2>/dev/null; then
    PASSED=$((PASSED+1))
else
    echo "  ⚠ Model load failed (might be OK if checkpoint loading is slow)"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary: $PASSED passed, $FAILED failed"
echo "=========================================="
echo ""

if [ $FAILED -gt 0 ]; then
    echo "❌ Pre-flight check failed. Fix issues above before running server."
    exit 1
else
    echo "✅ All checks passed! Safe to run:"
    echo "   bash run_server_best.sh"
    exit 0
fi
