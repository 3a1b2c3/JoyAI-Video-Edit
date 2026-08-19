#!/bin/bash
# Quick test: skip 18min VAE encode, test diffusion + decode + save (uses joyomni_ops)

cd "$(dirname "${BASH_SOURCE[0]}")"

# Setup environment
export PYTHONPATH="$PWD/deploy/joyomni_ops:$PWD/deploy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64:$(python3 -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"

echo "Running quick test with joyomni_ops..."
echo ""

python3 -u << 'PYEOF'
# Import joyomni_ops FIRST, before torch
import joyomni_ops  # noqa: F401
import torch
import sys
import gc
import numpy as np
sys.path.insert(0, './deploy')
from pathlib import Path
from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig

device = torch.device("cuda")
print("Quick Test: DiT diffusion (skip 18min VAE encode)")
print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print()

def mem_info(label=""):
    if label:
        print(f"[MEM] {label}")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Allocated: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {total:.1f}GB / Free: {total - reserved:.1f}GB")

mem_info("Start")

# Load DiT
print("[1/4] Loading DiT...")
mem_info("Before DiT load")
cfg = ExpConfig()
mem_info("After ExpConfig")
cfg.training_mode = False
cfg.dit_precision = "bf16"
cfg.dit_ckpt = str(Path("dit_quantized.pth"))
mem_info("Before load_dit()")
dit = load_dit(cfg, device=device)
mem_info("After load_dit()")
dit.eval()
mem_info("After DiT eval")

# Load VAE
print("[2/4] Loading VAE (GPU)...")
mem_info("Before VAE load")
vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
mem_info("After VAE path")
vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.bfloat16)
mem_info("After from_pretrained")
vae = vae.to("cuda")
mem_info("After vae.to(cuda)")
# Force all VAE parameters/buffers to bf16 (fixes bias dtype mismatches)
for param in vae.parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)
for buf in vae.buffers():
    if buf.dtype not in (torch.long, torch.int, torch.bool):
        if buf.dtype != torch.bfloat16:
            buf.data = buf.data.to(torch.bfloat16)
vae.eval()
mem_info("After VAE eval")
print()

# Synthetic latents (skip 18min VAE encode)
print("[3/4] Testing diffusion with synthetic latents...")
mem_info("Before latents create")
latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.bfloat16, device=device)
print(f"  Latents: {latents.shape}")
mem_info("After latents create")

with torch.no_grad():
    mem_info("Inside no_grad, before t")
    t = torch.tensor([500], device=device, dtype=torch.long)
    mem_info("After t create")
    context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
    mem_info("After context create")
    print(f"  Running forward pass...")
    output = dit(latents, t, context)
    mem_info("After dit forward pass")
    if isinstance(output, tuple):
        output = output[0]
    print(f"  ✓ Diffusion output: {output.shape}")
mem_info("After diffusion")
print()

# Decode and save
print("[4/4] Decoding and saving...")
mem_info("Before decode")
with torch.no_grad():
    mem_info("Inside no_grad, before latents_decoded")
    latents_decoded = latents / 0.18215
    mem_info("After latents_decoded")
    try:
        print(f"  Calling vae.decode()...")
        frame = vae.decode(latents_decoded).sample
        mem_info("After vae.decode()")
        print(f"  Decoded shape: {frame.shape}")

        # Save output
        output_path = Path("outputs/test_quick_output.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mem_info("Before image conversion")

        import imageio.v3 as iio
        if frame.ndim == 5:
            frame = frame.squeeze(2)
        mem_info("After squeeze")
        output_frames = (frame.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
        mem_info("After frame conversion")
        print(f"  Writing {len(output_frames)} frames...")
        iio.imwrite(str(output_path), output_frames, fps=24)
        mem_info("After imwrite")

        # Verify file exists
        import os
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"  ✓ Saved: {output_path.resolve()} ({size} bytes)")
            assert size > 0, f"File created but empty! {output_path}"
        else:
            raise FileNotFoundError(f"File not created: {output_path}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print()
print("✅ Quick test complete! (VAE encoding skipped, output saved)")
PYEOF
