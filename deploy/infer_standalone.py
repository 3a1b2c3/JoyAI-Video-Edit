#!/usr/bin/env python3
"""Standalone inference script (no server) for JoyAI-Video-Edit"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add deploy to path
sys.path.insert(0, str(Path(__file__).parent))


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_video_frame(video_path: str, height: int, width: int) -> torch.Tensor:
    """Load first frame from video"""
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


def run_v2v_generation(runtime, args, ref_image):
    """Drive the streaming V2V session over the whole input video and collect decoded frames."""
    from xvideo.serving.joyomni_streaming import StreamingSettings

    settings = StreamingSettings(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        seed=args.seed,
    )
    session = runtime.create_v2v_session(args.prompt, settings=settings, ref_image=ref_image)

    decoded = []  # list of (seq, rgb_frame)

    def _consume(results):
        for result in results:
            jpegs = result.jpegs or []
            metas = result.source_metas or []
            valid_count = result.valid_count if result.valid_count is not None else len(jpegs)
            for jpeg_bytes, meta in list(zip(jpegs, metas))[:valid_count]:
                bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded.append((meta.get("seq", len(decoded)), rgb))

    cap = cv2.VideoCapture(args.input_video)
    try:
        seq = 0
        while len(decoded) < args.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            seq += 1
            frame = cv2.resize(frame, (args.width, args.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = session.push_frame(
                Image.fromarray(frame), frame_meta={"seq": seq, "t_capture_ms": 0.0}
            )
            _consume(results)
    finally:
        cap.release()

    try:
        session.flush_pending()
        deadline = time.time() + 120.0
        misses = 0
        while misses < 20 and time.time() < deadline:
            result = session.wait_async_result(timeout=0.5)
            if result is not None:
                _consume([result])
                misses = 0
            else:
                misses += 1
    finally:
        session.close()

    if not decoded:
        raise RuntimeError("Generation produced no output frames")

    decoded.sort(key=lambda item: item[0])
    return np.stack([frame for _, frame in decoded[: args.num_frames]])


def main():
    args = parse_args()

    print("=" * 70)
    print("JoyAI-Video-Edit Standalone Inference")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load via JoyOmniRuntime
    print("[1/4] Loading models...")
    from xvideo.serving.joyomni_streaming import JoyOmniRuntime

    deploy_root = Path(__file__).parent
    runtime = JoyOmniRuntime.load(
        dit_ckpt=str(deploy_root / "deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"),
        vae_ckpt=str(deploy_root / "deps/checkpoints/JoyAI-Video-Edit/vae"),
        text_encoder_ckpt=str(deploy_root / "deps/checkpoints/MiMo-VL-7B-RL-2508"),
        device=device,
    )
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
    frames = run_v2v_generation(runtime, args, style_img)
    print(f"✓ Generated {frames.shape[0]} frames")
    print()

    # Save
    print("[4/4] Saving output...")
    import imageio.v3 as iio
    iio.imwrite(args.output, frames, fps=24)
    print(f"✓ Saved: {args.output}")

    print()
    print("=" * 70)
    print("✅ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
