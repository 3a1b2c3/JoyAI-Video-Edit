#!/usr/bin/env python3
"""Debug inference pipeline - trace intermediate outputs and quality"""

import os
import sys
import gc
from pathlib import Path

os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.getcwd()}/deploy:{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Must import joyomni_ops before torch
import joyomni_ops  # noqa: F401

import torch
import cv2
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio
import traceback

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image


def save_frame_debug(frames, step_name, index=0):
    """Save frame for visual inspection"""
    out_dir = Path("debug_frames")
    out_dir.mkdir(exist_ok=True)

    if frames.ndim == 5:
        frames = frames.squeeze(2)
    if frames.ndim == 4:
        frame = frames[index].permute(1, 2, 0)
    else:
        frame = frames[index]

    output = (frame.to(torch.float32) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
    path = out_dir / f"{step_name}.png"
    iio.imwrite(str(path), output)
    print(f"  📊 Saved debug frame: {path}")


def stats(tensor, name=""):
    """Print tensor statistics"""
    if name:
        print(f"  {name}:")
    print(f"    shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    print(f"    min: {tensor.min():.6f}, max: {tensor.max():.6f}, mean: {tensor.mean():.6f}, std: {tensor.std():.6f}")


print("=" * 70)
print("DEBUG: JoyAI-Video-Edit Inference Pipeline")
print("=" * 70)
print()

device = torch.device("cuda")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

seed_everything(42)

# Load models
print("[1/5] Loading models...")
cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"
cfg.dit_ckpt = str(Path("dit_quantized.pth"))
dit = load_dit(cfg, device=device)
dit.eval()
print("  ✓ DiT loaded")

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
print("  ✓ VAE loaded")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    local_files_only=False,
    torch_dtype=torch.float16,
    device_map="cpu"
).eval()
print("  ✓ Qwen2.5-VL loaded")
print()

# Load small video (1 frame only for debugging)
print("[2/5] Loading video...")
video_path = "assets/input.mp4"
cap = cv2.VideoCapture(video_path)
ret, bgr = cap.read()
cap.release()

if not ret:
    print(f"ERROR: Cannot read video {video_path}")
    sys.exit(1)

# Resize to 256x256
bgr = cv2.resize(bgr, (256, 256))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
frame = torch.from_numpy(rgb).float() / 255.0
frames_tensor = frame.unsqueeze(0).to(device, dtype=torch.bfloat16)
print(f"  ✓ Loaded 1 frame")
stats(frames_tensor, "Input frames")
print()

# VAE encode
print("[3/5] VAE encode...")
with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2).to("cpu")
    z = frames_chw[0:1].unsqueeze(2)
    print(f"  z shape: {z.shape}")
    posterior = vae.encode(z).latent_dist
    sample = posterior.sample() * 0.18215
    latents = sample.to(device)

stats(latents, "Encoded latents")
save_frame_debug(latents / 0.18215, "01_encoded_latents")
print()

# Style encoding
print("[4/5] Style image encoding...")
style_path = "assets/image.png"
style_img = Image.open(style_path).convert("RGB")
prompt_text = "Describe the visual style, colors, atmosphere, and artistic composition of this image.\n<|vision_start|><|image_pad|><|vision_end|>"

inputs = processor(text=prompt_text, images=[style_img], return_tensors="pt")
with torch.no_grad():
    outputs = qwen_model(**{k: v.to("cpu") for k, v in inputs.items()}, output_hidden_states=True)
    prompt_embeds = outputs.hidden_states[-1]

emb_dim = prompt_embeds.shape[-1]
if emb_dim != 4096:
    proj = torch.nn.Linear(emb_dim, 4096, dtype=prompt_embeds.dtype, device="cpu")
    with torch.no_grad():
        prompt_embeds = proj(prompt_embeds)

seq_len = prompt_embeds.shape[1]
if seq_len >= 256:
    context_style = prompt_embeds[:, :256, :]
else:
    pad_size = 256 - seq_len
    context_style = torch.cat([
        prompt_embeds,
        torch.zeros(1, pad_size, 4096, device=prompt_embeds.device)
    ], dim=1)

context_style = context_style.to(device).to(torch.bfloat16)
stats(context_style, "Style context")
print()

# Diffusion (just 3 steps for debug)
print("[5/5] Diffusion (3 steps for debug)...")
steps = 3
cfg_scale = 7.5

for step in range(steps):
    t = (steps - step - 1) / steps
    t_idx = int(t * 1000)
    t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

    context = context_style

    # Conditional
    output_cond = dit(latents, t_tensor, context)
    if isinstance(output_cond, (tuple, list)):
        output_cond = output_cond[0]

    # Unconditional
    context_uncond = torch.zeros_like(context)
    output_uncond = dit(latents, t_tensor, context_uncond)
    if isinstance(output_uncond, (tuple, list)):
        output_uncond = output_uncond[0]

    # CFG blend
    model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

    print(f"  Step {step}/{steps}:")
    print(f"    t={t:.3f}, t_idx={t_idx}")
    stats(output_cond, "    output_cond")
    stats(output_uncond, "    output_uncond")
    stats(model_output, "    CFG blend")

    sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
    latents = latents + model_output * (-sigma * 0.1)

    stats(latents, "    updated latents")
    print()

print("[DECODE] VAE decode...")
with torch.no_grad():
    latents_decoded = latents / 0.18215
    latents_decoded = latents_decoded.to("cpu")
    frame = vae.decode(latents_decoded).sample
    frame = frame.to(device)

stats(frame, "Decoded frame")
save_frame_debug(frame, "02_decoded_frame")
print()

# Save full comparison video
print("[SAVE] Saving comparison video...")
output_frames = (frame.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
iio.imwrite("debug_frames/03_final_output.mp4", output_frames, fps=24)
print(f"  ✓ Saved: debug_frames/03_final_output.mp4")
print()

print("=" * 70)
print("✅ Debug complete!")
print("=" * 70)
print()
print("Check debug_frames/ for:")
print("  - 01_encoded_latents.png: Raw VAE encoded output")
print("  - 02_decoded_frame.png: After diffusion + VAE decode")
print("  - 03_final_output.mp4: Full output video")
print()
print("Quality issues to check:")
print("  1. Are latents reasonable (not all zeros/NaN)?")
print("  2. Is diffusion output changing the latents?")
print("  3. Is CFG actually blending (unconditional + conditional)?")
print("  4. Is decoded output smooth or blocky/pixelated?")
