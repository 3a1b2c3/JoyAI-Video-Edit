#!/usr/bin/env python3
"""Standalone inference script (no server) for JoyAI-Video-Edit"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

# Add deploy to path
sys.path.insert(0, str(Path(__file__).parent))

from xvideo.models.models import load_model
from xvideo.models.pipeline import Pipeline
from xvideo.config import ExpConfig
from xvideo.models.scheduler import FlowMatchEulerScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="JoyAI standalone inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=120, help="Number of output frames")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=864, help="Output width")
    parser.add_argument("--num_steps", type=int, default=30, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dit_checkpoint", type=str,
                        default=os.environ.get("JOYOMNI_DIT_CHECKPOINT", "deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"),
                        help="DiT checkpoint path")
    parser.add_argument("--vae_checkpoint", type=str,
                        default=os.environ.get("JOYOMNI_VAE_CHECKPOINT", "deploy/deps/checkpoints/JoyAI-Video-Edit/vae"),
                        help="VAE checkpoint path")
    parser.add_argument("--text_encoder_checkpoint", type=str,
                        default=os.environ.get("JOYOMNI_TEXT_ENCODER_CHECKPOINT", "deploy/deps/checkpoints/MiMo-VL-7B-RL-2508"),
                        help="Text encoder checkpoint path")
    return parser.parse_args()


def load_video_frame(video_path: str, height: int, width: int) -> torch.Tensor:
    """Load first frame from video"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read video: {video_path}")

    # Resize and convert to tensor
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).float() / 255.0  # [H,W,3] in [0,1]
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return frame_tensor


def save_video(frames: torch.Tensor, output_path: str, fps: int = 24):
    """Save tensor as video"""
    import imageio.v3 as iio
    import numpy as np

    # Convert from [B,T,3,H,W] or [B,3,T,H,W] to [T,H,W,3]
    if frames.ndim == 5:
        if frames.shape[1] == 3:  # [B,3,T,H,W]
            frames = frames.permute(0, 2, 3, 4, 1)  # [B,T,H,W,3]
        frames = frames[0]  # Remove batch dim -> [T,H,W,3]

    frames = frames.detach().cpu().float()
    frames = (frames * 255).clamp(0, 255).numpy().astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    iio.imwrite(output_path, frames, fps=fps)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("JoyAI-Video-Edit Standalone Inference")
    print("=" * 70)
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load models
    print("[1/4] Loading models...")
    cfg = ExpConfig()
    model = load_model(cfg, device=device)
    pipeline = Pipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        transformer=model.transformer,
        scheduler=FlowMatchEulerScheduler(),
        args=cfg,
    )
    pipeline = pipeline.to(device)
    print(f"✓ Models loaded on {device}")
    print()

    # Load input
    print("[2/4] Loading input video...")
    input_frame = load_video_frame(args.input_video, args.height, args.width)
    print(f"✓ Loaded: {input_frame.shape}")
    print()

    # Generate
    print("[3/4] Generating...")
    with torch.no_grad():
        output = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            latents=input_frame,
        )
    print(f"✓ Generated: {output.shape}")
    print()

    # Save
    print("[4/4] Saving output...")
    save_video(output.images, args.output, fps=24)

    print()
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
