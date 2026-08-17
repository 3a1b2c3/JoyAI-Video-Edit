#!/bin/bash
# Quick test: skip VAE encode (18min!), test DiT diffusion only with synthetic latents

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"

python3 << 'PYEOF'
import torch
import sys
sys.path.insert(0, './deploy')
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig
from pathlib import Path

device = torch.device("cuda")
print("Quick Test: DiT diffusion (skip 18min VAE encode)")
print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print()

# Load DiT
print("[1/2] Loading DiT...")
cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "fp16"
cfg.dit_ckpt = str(Path("deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth"))
dit = load_dit(cfg, device=device)
dit.eval()
print(f"  ✓ DiT loaded ({torch.cuda.memory_allocated() / 1e9:.1f}GB)")
print()

# Synthetic latents (skip 18min VAE)
print("[2/2] Testing diffusion with synthetic latents...")
latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.float16, device=device)
print(f"  Latents: {latents.shape}")

with torch.no_grad():
    t = torch.tensor([500], device=device, dtype=torch.long)
    context = torch.randn(1, 256, 4096, dtype=torch.float16, device=device)
    print(f"  Running forward pass...")
    output = dit(latents, t, context)
    if isinstance(output, tuple):
        output = output[0]
    print(f"  ✓ Output: {output.shape}")

print()
print("✅ Quick test passed! (VAE encoding skipped)")
PYEOF
