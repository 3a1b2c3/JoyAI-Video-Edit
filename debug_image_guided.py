#!/usr/bin/env python3
"""Debug image-guided inference - trace style encoding and diffusion"""

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
from PIL import Image

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
print("DEBUG: Image-Guided Inference")
print("=" * 70)
print()

device = torch.device("cuda")
seed_everything(42)

# Load models
print("[1/6] Loading models...")
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
print("[2/6] Loading input video...")
cap = cv2.VideoCapture("assets/input.mp4")
ret, bgr = cap.read()
cap.release()

bgr = cv2.resize(bgr, (256, 256))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
frame = torch.from_numpy(rgb).float() / 255.0
frames_tensor = frame.unsqueeze(0).to(device, dtype=torch.bfloat16)

save_visual(frames_tensor, "00_input")
print(f"  Input: {frames_tensor.shape}")
print()

# Load style image
print("[3/6] Loading style image...")
style_path = "assets/image.png"
style_img = Image.open(style_path).convert("RGB")
save_visual(torch.tensor(np.array(style_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0, "01_style_image")
print(f"  Style: {style_path}")
print()

# Encode style image
print("[4/6] Encoding style image...")
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
print(f"  Context: {context_style.shape}")
print()

# VAE encode
print("[5/6] VAE encode...")
with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2).to("cpu")
    z = frames_chw[0:1].unsqueeze(2)
    posterior = vae.encode(z).latent_dist
    latents = posterior.sample() * 0.18215
    latents = latents.to(device)

save_visual(latents / 0.18215, "02_encoded")
print(f"  Latents: {latents.shape}")
print()

# Diffusion
print("[6/6] Diffusion (1 step)...")
t = 0.5
t_idx = int(t * 1000)
t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)
cfg_scale = 7.5

print("  Conditional pass (with style)...")
output_cond = dit(latents, t_tensor, context_style)
if isinstance(output_cond, (tuple, list)):
    output_cond = output_cond[0]
save_visual(output_cond, "03_diff_cond")
torch.cuda.empty_cache()
gc.collect()

print("  Unconditional pass (no style)...")
context_uncond = torch.zeros_like(context_style)
output_uncond = dit(latents, t_tensor, context_uncond)
if isinstance(output_uncond, (tuple, list)):
    output_uncond = output_uncond[0]
save_visual(output_uncond, "03_diff_uncond")
torch.cuda.empty_cache()
gc.collect()

print("  CFG blend...")
model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
save_visual(model_output, "03_diff_cfg")

sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
latents = latents + model_output * (-sigma * 0.1)
save_visual(latents, "04_diffused")

del output_cond, output_uncond, model_output, context_style, context_uncond
torch.cuda.empty_cache()
gc.collect()
print()

# Decode
print("  VAE decode...")
with torch.no_grad():
    latents_decoded = latents / 0.18215
    latents_decoded = latents_decoded.to("cpu")
    frame_out = vae.decode(latents_decoded).sample
    frame_out = frame_out.to(device)

if frame_out.ndim == 5:
    frame_out = frame_out.squeeze(2)

save_visual(frame_out, "05_decoded")
print()

# Save output
output_frames = (frame_out.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
iio.imwrite("debug_frames/06_output.mp4", output_frames, fps=24)
print()

print("=" * 70)
print("✅ Complete! Check debug_frames/:")
print("  00_input.png         - Input frame")
print("  01_style_image.png   - Reference style image")
print("  02_encoded.png       - VAE encoded latents")
print("  03_diff_*.png        - Diffusion outputs (cond/uncond/cfg)")
print("  04_diffused.png      - Latents after diffusion")
print("  05_decoded.png       - Final frame")
print("  06_output.mp4        - Final video")
print("=" * 70)
