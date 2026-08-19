#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/deploy/joyomni_ops:$SCRIPT_DIR/deploy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64:$(python3 -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$SCRIPT_DIR/.cache/torch"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

echo "=========================================="
echo "DiT Model Loading Test (quantized + joyomni_ops)"
echo "=========================================="
echo ""

python3 -u << 'PYEOF'
# Import joyomni_ops FIRST, before torch
import joyomni_ops  # noqa: F401

import sys
import torch
from pathlib import Path

sys.path.insert(0, './deploy')
from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

device = torch.device("cuda")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print()

seed_everything(42)

cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"
cfg.dit_ckpt = str(Path("dit_quantized.pth"))  # Use quantized (2x smaller) checkpoint

print("[1/3] Loading DiT...")
mem_before = torch.cuda.memory_allocated() / 1e9
print(f"  Memory before: {mem_before:.1f}GB")

dit = load_dit(cfg, device=device)

mem_after = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"  Memory after: {mem_after:.1f}GB (+{mem_after - mem_before:.1f}GB)")
print(f"  Memory reserved: {mem_reserved:.1f}GB")
print()

print("[2/3] Model info")
model_dtype = next(dit.parameters()).dtype
model_device = next(dit.parameters()).device
total_params = sum(p.numel() for p in dit.parameters())
print(f"  Dtype: {model_dtype}")
print(f"  Device: {model_device}")
print(f"  Parameters: {total_params / 1e9:.2f}B")
print()

print("[3/3] Memory summary")
print(f"  Allocated: {mem_after:.1f}GB")
print(f"  Reserved:  {mem_reserved:.1f}GB")
print(f"  Total GPU: {mem_total:.1f}GB")
print(f"  Free:      {mem_total - mem_reserved:.1f}GB")
print()

if mem_total - mem_reserved < 5:
    print("⚠ WARNING: Less than 5GB free on GPU")
else:
    print("✅ Enough GPU memory for VAE")

PYEOF

echo ""
echo "=========================================="
echo "✅ Model loaded successfully"
echo "=========================================="
