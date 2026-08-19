#!/usr/bin/env python3
"""Debug inference - save visual outputs at each step"""

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
import imageio.v3 as iio
import traceback

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything


def save_visual(tensor, name, step_name):
    """Save tensor as PNG for visual inspection"""
    out_dir = Path("debug_frames")
    out_dir.mkdir(exist_ok=True)

    if tensor.ndim == 5:
        tensor = tensor.squeeze(2)
    if tensor.ndim == 4:
        frame = tensor[0].permute(1, 2, 0)
    else:
        frame = tensor[0]

    # Normalize to 0-255 (convert to float32 for numpy compatibility)
    frame = frame.to(torch.float32)
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max > frame_min:
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
    else:
        frame_norm = torch.zeros_like(frame)

    output = (frame_norm * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    if output.shape[-1] == 1:
        output = np.repeat(output, 3, axis=-1)

    path = out_dir / f"{step_name}_{name}.png"
    iio.imwrite(str(path), output)
    print(f"  💾 {path} (min={frame_min:.3f}, max={frame_max:.3f})")


print("=" * 70)
print("DEBUG: Visual Pipeline Trace")
print("=" * 70)
print()

device = torch.device("cuda")
seed_everything(42)

print("[1/5] Loading models...")
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

print("[2/5] Loading video...")
video_path = "assets/input.mp4"
cap = cv2.VideoCapture(video_path)
ret, bgr = cap.read()
cap.release()

if not ret:
    print(f"ERROR: Cannot read {video_path}")
    sys.exit(1)

bgr = cv2.resize(bgr, (256, 256))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
frame = torch.from_numpy(rgb).float() / 255.0
frames_tensor = frame.unsqueeze(0).to(device, dtype=torch.bfloat16)
print(f"✓ Loaded frame {frames_tensor.shape}")
print()

print("[3/5] VAE encode...")
Path("debug_frames").mkdir(exist_ok=True)
with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2).to("cpu")
    z = frames_chw[0:1].unsqueeze(2)
    posterior = vae.encode(z).latent_dist
    latents = posterior.sample() * 0.18215
    latents = latents.to(device)

save_visual(latents / 0.18215, "encoded", "01")
print()

print("[4/5] Diffusion (1 step - avoid OOM)...")
context_style = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
cfg_scale = 7.5

t = 0.5
t_idx = int(t * 1000)
t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

print("  Conditional pass...")
output_cond = dit(latents, t_tensor, context_style)
if isinstance(output_cond, (tuple, list)):
    output_cond = output_cond[0]
save_visual(output_cond, "diff_cond", "02")
torch.cuda.empty_cache()
gc.collect()

print("  Unconditional pass...")
context_uncond = torch.zeros_like(context_style)
output_uncond = dit(latents, t_tensor, context_uncond)
if isinstance(output_uncond, (tuple, list)):
    output_uncond = output_uncond[0]
save_visual(output_uncond, "diff_uncond", "02")
torch.cuda.empty_cache()
gc.collect()

model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
save_visual(model_output, "diff_cfg", "02")

sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
latents = latents + model_output * (-sigma * 0.1)
save_visual(latents, "diffused", "03")

del output_cond, output_uncond, model_output, context_style, context_uncond
torch.cuda.empty_cache()
gc.collect()
print()

print("[5/5] VAE decode...")
with torch.no_grad():
    latents_decoded = latents / 0.18215
    latents_decoded = latents_decoded.to("cpu")
    frame_out = vae.decode(latents_decoded).sample
    frame_out = frame_out.to(device)

save_visual(frame_out, "decoded", "04")
print()

# Save as MP4
output_frames = (frame_out.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
iio.imwrite("debug_frames/05_output.mp4", output_frames, fps=24)
print("✓ Saved output.mp4")
print()

print("=" * 70)
print("✅ Complete! Check debug_frames/:")
print("  01_*_encoded.png     - Input after VAE encode (before diffusion)")
print("  02_*_diff_*.png      - Diffusion outputs (cond/uncond/cfg)")
print("  03_*_diffused.png    - Latents after diffusion")
print("  04_*_decoded.png     - Final frame after VAE decode")
print("  05_output.mp4        - Final video")
print("=" * 70)
