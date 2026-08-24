#!/usr/bin/env python3
"""Standalone inference script (no server) for JoyAI-Video-Edit"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# Add deploy to path
sys.path.insert(0, str(Path(__file__).parent))

from xvideo.serving.joyomni_streaming import JoyOmniRuntime
from xvideo.config import ExpConfig


def parse_args():
    parser = argparse.ArgumentParser(description="JoyAI standalone inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--image", type=str, default=None, help="Optional style/reference image for conditioning")
    parser.add_argument("--num_frames", type=int, default=120, help="Number of output frames")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=864, help="Output width")
    parser.add_argument("--num_steps", type=int, default=30, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    frame_tensor = torch.from_numpy(frame).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    return frame_tensor


def main():
    args = parse_args()

    print("=" * 70)
    print("JoyAI-Video-Edit Standalone Inference")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load runtime (same as server uses)
    print("[1/4] Loading models...")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_ckpt = os.environ.get("JOYOMNI_DIT_CHECKPOINT", "deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth")
    cfg.vae_arch_config['pretrained'] = os.environ.get("JOYOMNI_VAE_CHECKPOINT", "deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
    cfg.text_encoder_arch_config['params']['text_encoder_ckpt'] = os.environ.get("JOYOMNI_TEXT_ENCODER_CHECKPOINT", "deploy/deps/checkpoints/MiMo-VL-7B-RL-2508")

    runtime = JoyOmniRuntime.load(cfg, device=device)
    print(f"✓ Models loaded on {device}")
    print()

    # Load input
    print("[2/4] Loading input video...")
    input_frame = load_video_frame(args.input_video, args.height, args.width)
    print(f"✓ Loaded: {input_frame.shape}")
    print()

    # Load style image if provided
    if args.image:
        print("[3/4] Loading style image...")
        if not os.path.exists(args.image):
            print(f"ERROR: Style image not found: {args.image}")
            return 1
        style_img = Image.open(args.image).convert("RGB")
        print(f"✓ Loaded: {style_img.size}")
        print()
        step_num = 4
    else:
        style_img = None
        step_num = 3

    # Generate
    print(f"[{step_num}/4] Generating...")
    with torch.no_grad():
        output = runtime.infer(
            prompt=args.prompt,
            image=style_img,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    print(f"✓ Generated")
    print()

    # Save
    print("[4/4] Saving output...")
    output.save(args.output)

    print()
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
