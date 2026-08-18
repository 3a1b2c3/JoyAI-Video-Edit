#!/usr/bin/env python3
"""Quick test: skip VAE encode, test DiT diffusion with synthetic latents"""

import torch
import sys
import os
import gc
import numpy as np
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig

device = torch.device("cuda")
print("Quick Test: DiT diffusion (skip 18min VAE encode)")
print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def show_mem(label):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[MEM] {label}: alloc={alloc:.1f}GB reserved={reserved:.1f}GB free={total-reserved:.1f}GB")

show_mem("Start")
print()

# Load DiT
print("[1/4] Loading DiT...")
show_mem("Before load_dit")
cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"
cfg.dit_ckpt = str(Path("dit_quantized.pth"))  # Use quantized checkpoint
dit = load_dit(cfg, device=device)
show_mem("After load_dit")
dit.eval()
show_mem("After dit.eval()")
print()

# Load VAE with bf16 dtype
print("[2/4] Loading VAE (bf16)...")
show_mem("Before VAE load")
vae = XVAEChunkCausal.from_pretrained(
    str(Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")),
    torch_dtype=torch.bfloat16
)
vae = vae.to(device)
# Force all VAE parameters/buffers to bf16 (fixes bias dtype mismatches)
for param in vae.parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)
for buf in vae.buffers():
    if buf.dtype not in (torch.long, torch.int, torch.bool):
        if buf.dtype != torch.bfloat16:
            buf.data = buf.data.to(torch.bfloat16)
vae.eval()
show_mem("After VAE load")
print()

# Synthetic latents (skip 18min VAE encode)
print("[3/4] Testing diffusion with synthetic latents...")
show_mem("Before latents create")
latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.bfloat16, device=device)
print(f"  Latents: {latents.shape}")
show_mem("Before diffusion forward")

with torch.no_grad():
    t = torch.tensor([500], device=device, dtype=torch.long)
    context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
    print(f"  Running forward pass...")
    output = dit(latents, t, context)
    show_mem("After diffusion forward")
    if isinstance(output, tuple):
        output = output[0]
    print(f"  ✓ Diffusion output: {output.shape}")
print()

# Decode and save
print("[4/4] Decoding and saving...")
show_mem("Before decode")
with torch.no_grad():
    latents_decoded = latents / 0.18215
    try:
        print(f"  Calling vae.decode()...")
        frame = vae.decode(latents_decoded).sample
        show_mem("After decode")
        print(f"  Decoded shape: {frame.shape}")

        # Save output
        output_path = Path("outputs/test_quick_output.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir: {output_path.parent.resolve()}")
        print(f"  Output path: {output_path.resolve()}")

        import imageio.v3 as iio
        if frame.ndim == 5:
            frame = frame.squeeze(2)
        print(f"  Frame shape before save: {frame.shape}, dtype: {frame.dtype}")
        output_frames = (frame.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
        print(f"  Output frames shape: {output_frames.shape}, dtype: {output_frames.dtype}")
        print(f"  Saving with imageio.v3...")
        iio.imwrite(str(output_path), output_frames, fps=24)

        # Verify file exists and has content
        import os
        assert os.path.exists(output_path), f"❌ ASSERT FAILED: File NOT created: {output_path.resolve()}"
        file_size = os.path.getsize(output_path)
        assert file_size > 0, f"❌ ASSERT FAILED: File empty (0 bytes): {output_path.resolve()}"
        print(f"  ✓ Saved: {output_path.resolve()} ({file_size} bytes)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print()
print("✅ Quick test complete! (VAE encoding skipped, output saved)")
