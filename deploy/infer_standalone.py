#!/usr/bin/env python3
"""Standalone inference for JoyAI Video Edit using JoyOmniRuntime."""

import argparse
import sys
import time
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvideo.config import ExpConfig
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings


def load_video(video_path: str, height: int = 480, width: int = 840) -> list[Image.Image]:
    """Load video frames as PIL images."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def save_video(frames: list[np.ndarray], output_path: str, fps: int = 24) -> None:
    """Save frames to video file."""
    if not frames:
        raise ValueError("No frames to save")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame.astype(np.uint8))

    out.release()


def infer(
    prompt: str,
    input_video: str,
    output_path: str,
    image_path: str | None = None,
    device: str = "cuda:0",
    height: int = 480,
    width: int = 840,
    fps: int = 24,
    num_inference_steps: int = 25,
) -> None:
    """Run inference using JoyOmniRuntime."""
    device_obj = torch.device(device)
    cfg = ExpConfig()

    print("Loading JoyOmniRuntime...")
    runtime = JoyOmniRuntime.load(cfg, device=device_obj)

    print(f"Loading input video: {input_video}")
    frames = load_video(input_video, height=height, width=width)

    if not frames:
        raise ValueError(f"No frames loaded from {input_video}")

    print(f"Loaded {len(frames)} frames")

    # Load reference image if provided
    ref_image = None
    if image_path:
        print(f"Loading reference image: {image_path}")
        ref_image = Image.open(image_path).convert("RGB")
        ref_image = ref_image.resize((width, height))

    print(f"Running inference with prompt: {prompt}")
    print(f"Frames: {len(frames)}, Steps: {num_inference_steps}")

    # Create streaming session
    settings = StreamingSettings(
        num_inference_steps=num_inference_steps,
        ref_image=ref_image,
        prompt=prompt,
    )
    session = runtime.new_session(settings=settings, name="standalone")

    # Push all frames to session
    output_frames = []
    frame_count = len(frames)

    try:
        for i, frame in enumerate(frames):
            if (i + 1) % 50 == 0 or i == frame_count - 1:
                print(f"  Processing frame {i + 1}/{frame_count}")

            results = session.push_frame(frame, frame_meta={"seq": i + 1})

            # Collect decoded output frames
            for result in results:
                if result.decoded_pixels is not None:
                    pixels = result.decoded_pixels
                    if isinstance(pixels, torch.Tensor):
                        pixels = pixels.cpu().numpy()
                    if pixels.dtype != np.uint8:
                        pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
                    if pixels.shape[0] == 3:
                        pixels = np.transpose(pixels, (1, 2, 0))
                    output_frames.append(pixels)

        # Finalize session
        final_results = session.finalize()
        for result in final_results:
            if result.decoded_pixels is not None:
                pixels = result.decoded_pixels
                if isinstance(pixels, torch.Tensor):
                    pixels = pixels.cpu().numpy()
                if pixels.dtype != np.uint8:
                    pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
                if pixels.shape[0] == 3:
                    pixels = np.transpose(pixels, (1, 2, 0))
                output_frames.append(pixels)

        if not output_frames:
            raise RuntimeError("No output frames generated")

        print(f"Saving {len(output_frames)} frames to: {output_path}")
        save_video(output_frames, output_path, fps=fps)
        print("Done!")

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone JoyAI Video Edit inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for style/content")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path (MP4/WebM)")
    parser.add_argument("--output", type=str, help="Output video path (default: output_<timestamp>.mp4)")
    parser.add_argument("--image", type=str, help="Reference/style image for conditioning")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (default: cuda:0)")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=840, help="Output video width")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")

    args = parser.parse_args()
    output_path = args.output or f"output_{int(time.time())}.mp4"

    infer(
        prompt=args.prompt,
        input_video=args.input_video,
        output_path=output_path,
        image_path=args.image,
        device=args.device,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main()
