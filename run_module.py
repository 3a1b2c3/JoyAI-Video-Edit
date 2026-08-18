#!/usr/bin/env python3
"""Full inference runner - load video, encode, diffuse, decode, save."""

import os
import sys
import torch
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.environ.get('PYTHONPATH', '')}"

from inference import full_inference

if __name__ == "__main__":
    device = torch.device("cuda")

    # Parse arguments: video_path [output_path] [prompt] [num_frames] [height] [width] [steps]
    video_path = sys.argv[1] if len(sys.argv) > 1 else "assets/Recording 2026-08-12 205529.mp4"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/dit_output.mp4"
    prompt = sys.argv[3] if len(sys.argv) > 3 else None
    num_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    height = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    width = int(sys.argv[6]) if len(sys.argv) > 6 else 256
    steps = int(sys.argv[7]) if len(sys.argv) > 7 else 1

    if not prompt:
        print("⚠️  WARNING: No prompt provided - output will be noise!")
        print("Usage: python run_module.py video.mp4 output.mp4 'your prompt' [frames] [height] [width] [steps]")
        print()

    full_inference(
        video_path=video_path,
        output_path=output_path,
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        steps=steps,
        device=device
    )
