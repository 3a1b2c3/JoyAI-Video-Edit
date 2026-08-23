#!/usr/bin/env python3
"""Debug: Compare random context vs text-encoded context"""

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

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def save_visual(tensor, name):
    """Save tensor as PNG"""
    out_dir = Path("debug_frames")
    out_dir.mkdir(exist_ok=True)

    if tensor.ndim == 5:
        tensor = tensor.squeeze(2)
    if tensor.ndim == 4:
        frame = tensor[0].permute(1, 2, 0)
    else:
        frame = tensor[0]

    frame = frame.to(torch.float32)
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max > frame_min:
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
    else:
        frame_norm = torch.zeros_like(frame)

    output = (frame_norm * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    if output.shape[-1] == 1:
        output = np.repeat(output, 3, axis=-1)

    path = out_dir / f"{name}.png"
    iio.imwrite(str(path), output)
    print(f"  💾 {path}")


print("=" * 70)
print("DEBUG: Text Encoding vs Random Context")
print("=" * 70)
print()

device = torch.device("cuda")
seed_everything(42)

# Load models
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

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    local_files_only=False,
    torch_dtype=torch.float16,
    device_map="cpu"
).eval()

print("✓ Models loaded")
print()

# Load input video
print("[2/5] Loading input video...")
cap = cv2.VideoCapture("assets/input.mp4")
ret, bgr = cap.read()
cap.release()

bgr = cv2.resize(bgr, (256, 256))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
frame = torch.from_numpy(rgb).float() / 255.0
frames_tensor = frame.unsqueeze(0).to(device, dtype=torch.bfloat16)
print("✓ Loaded")
print()

# VAE encode
print("[3/5] VAE encode...")
with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2).to("cpu")
    z = frames_chw[0:1].unsqueeze(2)
    posterior = vae.encode(z).latent_dist
    latents_base = posterior.sample() * 0.18215
    latents_base = latents_base.to(device)

print("✓ Encoded")
print()

# Test 1: Random context (current approach)
print("[4/5] Test 1: Random context (CURRENT - not working)...")
latents = latents_base.clone()
context_random = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)

t = 0.5
t_idx = int(t * 1000)
t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

output_cond = dit(latents, t_tensor, context_random)
if isinstance(output_cond, (tuple, list)):
    output_cond = output_cond[0]
save_visual(output_cond, "a_random_context_output")

sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
latents = latents + output_cond * (-sigma * 0.1)
save_visual(latents, "a_random_context_diffused")

del output_cond
torch.cuda.empty_cache()
gc.collect()
print()

# Test 2: Text-encoded context
print("[5/5] Test 2: Text-encoded context (FIX)...")
latents = latents_base.clone()

# Encode text prompt
prompt_text = "watercolor painting"
print(f"  Encoding text: '{prompt_text}'")

# Use Qwen to encode text (just text, no image)
inputs = processor(text=prompt_text, return_tensors="pt")

with torch.no_grad():
    outputs = qwen_model(**{k: v.to("cpu") for k, v in inputs.items()}, output_hidden_states=True)
    prompt_embeds = outputs.hidden_states[-1]

emb_dim = prompt_embeds.shape[-1]
print(f"  Raw embedding: {prompt_embeds.shape}, dim={emb_dim}")

# Project to 4096 if needed
if emb_dim != 4096:
    proj = torch.nn.Linear(emb_dim, 4096, dtype=prompt_embeds.dtype, device="cpu")
    with torch.no_grad():
        prompt_embeds = proj(prompt_embeds)

# Pad/trim to 256
seq_len = prompt_embeds.shape[1]
if seq_len >= 256:
    context_text = prompt_embeds[:, :256, :]
else:
    pad_size = 256 - seq_len
    context_text = torch.cat([
        prompt_embeds,
        torch.zeros(1, pad_size, 4096, device=prompt_embeds.device)
    ], dim=1)

context_text = context_text.to(device).to(torch.bfloat16)
print(f"  Final context: {context_text.shape}")
print(f"  Context stats: min={context_text.min():.3f}, max={context_text.max():.3f}, mean={context_text.mean():.3f}")

output_cond = dit(latents, t_tensor, context_text)
if isinstance(output_cond, (tuple, list)):
    output_cond = output_cond[0]
save_visual(output_cond, "b_text_encoded_output")

sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
latents = latents + output_cond * (-sigma * 0.1)
save_visual(latents, "b_text_encoded_diffused")

del output_cond, context_text, context_random
torch.cuda.empty_cache()
gc.collect()
print()

print("=" * 70)
print("✅ Complete! Check debug_frames/:")
print("  a_random_context_*.png     - Current approach (random)")
print("  b_text_encoded_*.png       - With text encoding")
print()
print("Compare outputs:")
print("  - If a_* and b_* look identical → Text encoding isn't helping")
print("  - If b_* looks different → Text encoding is working")
print("=" * 70)
