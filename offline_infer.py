#!/usr/bin/env python3
"""Offline JoyAI inference with custom video and prompt."""

import sys
from pathlib import Path

deploy_dir = Path(__file__).parent / "deploy"
sys.path.insert(0, str(deploy_dir))

import torch
import cv2
from PIL import Image

# Config
prompt_file = Path("C:/workspace/world/JoyAI-Video-Edit/assets/decart_prompt.txt")
video_file = Path("C:/workspace/world/JoyAI-Video-Edit/assets/Recording 2026-08-12 205529.mp4")
output_file = Path("outputs/offline_edited.mp4")

print("=" * 70)
print("Offline JoyAI Inference")
print("=" * 70)

# Load prompt
if prompt_file.exists():
    with open(prompt_file) as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt[:80]}...")
else:
    print(f"ERROR: Prompt file not found: {prompt_file}")
    sys.exit(1)

# Load video
if video_file.exists():
    print(f"Video: {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Frames: {frame_count} @ {fps:.1f} fps")
    cap.release()
else:
    print(f"ERROR: Video file not found: {video_file}")
    sys.exit(1)

# Check GPU
print(f"GPU: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print()

# Try to load models
print("Loading models...")

# Check DiT
dit_path = deploy_dir / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "dit" / "joyai_video_edit_dit_0811.pth"
if dit_path.exists():
    print(f"  ✓ DiT: {dit_path.stat().st_size / 1e9:.1f} GB")
else:
    print(f"  ✗ DiT not found: {dit_path}")
    print("  Download required: python download.py")
    sys.exit(1)

# Load VAE
try:
    from diffusers import AutoencoderKL
    vae_path = deploy_dir / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "vae"
    vae = AutoencoderKL.from_pretrained(str(vae_path), low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    print(f"  ✓ VAE loaded")
except Exception as e:
    print(f"  ✗ VAE error: {e}")
    sys.exit(1)

print()
print("Ready to process. DiT inference would:")
print("  1. Load video frames")
print("  2. Encode with VAE")
print("  3. Run 8-step diffusion")
print("  4. Decode output")
print("  5. Save MP4")
print()
print("DiT required to proceed.")
