#!/usr/bin/env python3
"""Check if conditioning is working - compare cond vs uncond outputs"""

import os
import sys
import gc
from pathlib import Path

os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.getcwd()}/deploy:{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import joyomni_ops  # noqa: F401

import torch
import cv2
import numpy as np

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything

print("=" * 70)
print("CHECK: Is Conditioning Working?")
print("=" * 70)
print()

device = torch.device("cuda")
seed_everything(42)

print("[1/3] Loading models...")
cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"
cfg.dit_ckpt = str(Path("dit_quantized.pth"))
dit = load_dit(cfg, device=device)
dit.eval()

vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.bfloat16)
vae = vae.to("cpu").eval()
for param in vae.parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)
for buf in vae.buffers():
    if buf.dtype not in (torch.long, torch.int, torch.bool):
        if buf.dtype != torch.bfloat16:
            buf.data = buf.data.to(torch.bfloat16)

print("✓ Models loaded")
print()

# Load input
print("[2/3] Loading input...")
cap = cv2.VideoCapture("assets/input.mp4")
ret, bgr = cap.read()
cap.release()

bgr = cv2.resize(bgr, (256, 256))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
frame = torch.from_numpy(rgb).float() / 255.0
frames_tensor = frame.unsqueeze(0).to(device, dtype=torch.bfloat16)

with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2).to("cpu")
    z = frames_chw[0:1].unsqueeze(2)
    posterior = vae.encode(z).latent_dist
    latents = posterior.sample() * 0.18215
    latents = latents.to(device)

print("✓ Input loaded and encoded")
print()

# Run diffusion
print("[3/3] Diffusion comparison...")
print()

t = 0.5
t_idx = int(t * 1000)
t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

# Two different contexts
context_a = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
context_b = torch.zeros_like(context_a)  # All zeros

print("Running forward pass with Context A (random)...")
output_a = dit(latents, t_tensor, context_a)
if isinstance(output_a, (tuple, list)):
    output_a = output_a[0]

torch.cuda.empty_cache()
gc.collect()

print("Running forward pass with Context B (zeros)...")
output_b = dit(latents, t_tensor, context_b)
if isinstance(output_b, (tuple, list)):
    output_b = output_b[0]

torch.cuda.empty_cache()
gc.collect()

# Compare
print()
print("=" * 70)
print("RESULTS:")
print("=" * 70)

# Statistical comparison
diff = (output_a - output_b).abs()
print()
print("Difference between outputs (Context A vs Context B):")
print(f"  Mean absolute difference: {diff.mean():.6f}")
print(f"  Max absolute difference:  {diff.max():.6f}")
print(f"  Min absolute difference:  {diff.min():.6f}")
print()

# Output statistics
print("Output A (random context):")
print(f"  shape: {output_a.shape}")
print(f"  mean: {output_a.mean():.6f}, std: {output_a.std():.6f}")
print(f"  min: {output_a.min():.6f}, max: {output_a.max():.6f}")
print()

print("Output B (zero context):")
print(f"  shape: {output_b.shape}")
print(f"  mean: {output_b.mean():.6f}, std: {output_b.std():.6f}")
print(f"  min: {output_b.min():.6f}, max: {output_b.max():.6f}")
print()

# Verdict
print("=" * 70)
if diff.mean() < 0.001:
    print("❌ VERDICT: Conditioning is NOT working!")
    print()
    print("The model produces nearly identical outputs regardless of context.")
    print("This means:")
    print("  - Image-guided mode won't work properly")
    print("  - Text prompts won't work")
    print("  - The model may not have conditioning layers enabled")
    print()
    print("Next steps:")
    print("  1. Check if the checkpoint includes condition modules")
    print("  2. Verify the model architecture expects conditioning")
    print("  3. Check if there's a separate text encoder in the codebase")
elif diff.mean() < 0.01:
    print("⚠️ VERDICT: Conditioning is WEAK")
    print()
    print("The model shows some difference but it's very small.")
    print("Guidance will be subtle.")
    print()
    print("Next steps:")
    print("  1. Increase CFG scale (try 15-20)")
    print("  2. Increase diffusion steps (try 50)")
    print("  3. Check if context encoding is correct")
else:
    print("✅ VERDICT: Conditioning IS working!")
    print()
    print("The model responds to different contexts.")
    print("Guidance should be effective with proper context encoding.")
print("=" * 70)
