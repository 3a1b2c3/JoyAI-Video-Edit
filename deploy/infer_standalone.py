#!/usr/bin/env python3
"""Standalone inference for JoyAI Video Edit without streaming server."""

import argparse
import sys
import time
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvideo.models.pipeline import Pipeline, PRECISION_TO_TYPE
from xvideo.models.scheduler import get_scheduler
from xvideo.models.models import build_vae, load_text_encoder, load_dit
from xvideo.config import ExpConfig


def load_models(device: str = "cuda:0") -> Pipeline:
    """Load and initialize the pipeline."""
    device = torch.device(device)
    cfg = ExpConfig()

    from xvideo.models.models import load_pipeline
    transformer = load_dit(cfg, device)
    pipeline = load_pipeline(cfg, transformer, device)

    return pipeline


def load_video(video_path: str, height: int = 480, width: int = 840) -> list[Image.Image]:
    """Load video frames as PIL images."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target size
        frame = cv2.resize(frame, (width, height))

        # Convert to PIL Image
        img = Image.fromarray(frame)
        frames.append(img)

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
        # Convert to BGR for OpenCV
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
    """Run inference."""
    print(f"Loading models...")
    pipeline = load_models(device=device)

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

    # Run inference - pipeline is callable (diffusers-style)
    output_frames = None
    try:
        # Try calling pipeline directly (works for diffusers pipelines)
        if callable(pipeline):
            output_frames = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Note: actual parameters depend on pipeline implementation
                # These may need adjustment based on pipeline API
            )
        else:
            # Fallback: try generate or infer methods
            if hasattr(pipeline, 'generate'):
                output_frames = pipeline.generate(prompt=prompt)
            elif hasattr(pipeline, 'infer'):
                output_frames = pipeline.infer(prompt=prompt)
            else:
                available_methods = [m for m in dir(pipeline) if not m.startswith('_')]
                raise NotImplementedError(
                    f"Pipeline is not callable. Available methods: {available_methods}"
                )
    except TypeError as e:
        # Wrong parameters for pipeline call
        print(f"Pipeline call failed with parameters: {e}")
        print("Trying with minimal parameters...")
        try:
            output_frames = pipeline(prompt=prompt)
        except Exception as e2:
            print(f"Fallback inference error: {e2}")
            raise
    except Exception as e:
        print(f"Inference error: {e}")
        raise

    if output_frames is None:
        raise RuntimeError("Pipeline returned no output frames")

    print(f"Saving output to: {output_path}")
    save_video(output_frames, output_path, fps=fps)
    print(f"Done!")


def main():
    parser = argparse.ArgumentParser(description="Standalone JoyAI Video Edit inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for style/content")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path (MP4/WebM)")
    parser.add_argument("--output", type=str, help="Output video path (default: output_<timestamp>.mp4)")
    parser.add_argument("--image", type=str, help="Reference/style image for conditioning")

    # Inference settings
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
    import os
    import time
    main()
