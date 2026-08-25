#!/usr/bin/env python3
"""Standalone inference for JoyAI Video Edit without streaming server."""

import argparse
import sys
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xvideo.models.pipeline import Pipeline
from xvideo.models.scheduler import get_scheduler
from xvideo.models.models import build_vae, load_text_encoder, load_dit
from xvideo.config import ExpConfig
from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings


def load_models(
    dit_ckpt: str,
    vae_ckpt: str,
    text_encoder_ckpt: str,
    device: str = "cuda:0",
) -> Pipeline:
    """Load and initialize the pipeline."""
    device = torch.device(device)

    # Load config
    cfg = ExpConfig()

    # Build and load VAE
    vae = build_vae(cfg, device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
    vae = vae.to(device).eval()
    vae.requires_grad_(False)

    # Load text encoder
    text_encoder = load_text_encoder(cfg, device)

    # Load DiT
    transformer = load_dit(cfg, device)
    transformer.load_state_dict(torch.load(dit_ckpt, map_location=device, weights_only=True))
    transformer = transformer.to(device).eval()
    transformer.requires_grad_(False)

    # Get scheduler
    scheduler = get_scheduler(cfg)

    # Create pipeline
    pipeline = Pipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=text_encoder.tokenizer,
        transformer=transformer,
        scheduler=scheduler,
        args=cfg,
    )
    pipeline = pipeline.to(device)

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
    dit_ckpt: str | None = None,
    vae_ckpt: str | None = None,
    text_encoder_ckpt: str | None = None,
    device: str = "cuda:0",
    height: int = 480,
    width: int = 840,
    fps: int = 24,
    num_inference_steps: int = 25,
) -> None:
    """Run inference."""
    print(f"Loading models from checkpoints...")
    pipeline = load_models(
        dit_ckpt=dit_ckpt,
        vae_ckpt=vae_ckpt,
        text_encoder_ckpt=text_encoder_ckpt,
        device=device,
    )

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

    # Run inference (this is a simplified version - actual implementation depends on pipeline API)
    # The pipeline might have a __call__ method or other inference methods
    try:
        if hasattr(pipeline, '__call__'):
            output_frames = pipeline(
                prompt=prompt,
                video_frames=frames,
                ref_image=ref_image,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
            )
        else:
            raise NotImplementedError("Pipeline inference method not found")
    except Exception as e:
        print(f"Inference error: {e}")
        print("Note: This is a skeleton implementation - actual inference requires the full pipeline API")
        raise

    print(f"Saving output to: {output_path}")
    save_video(output_frames, output_path, fps=fps)
    print(f"Done!")


def main():
    parser = argparse.ArgumentParser(description="Standalone JoyAI Video Edit inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for style/content")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path (MP4/WebM)")
    parser.add_argument("--output", type=str, help="Output video path (default: output_<timestamp>.mp4)")
    parser.add_argument("--image", type=str, help="Reference/style image for conditioning")

    # Model paths
    parser.add_argument("--dit_ckpt", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--text_encoder_ckpt", type=str, default=None, help="Text encoder checkpoint path")

    # Inference settings
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (default: cuda:0)")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    parser.add_argument("--width", type=int, default=840, help="Output video width")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")

    args = parser.parse_args()

    # Set default checkpoint paths from environment if not provided
    dit_ckpt = args.dit_ckpt or Path(os.environ.get("JOYOMNI_DIT_CHECKPOINT", ""))
    vae_ckpt = args.vae_ckpt or Path(os.environ.get("JOYOMNI_VAE_CHECKPOINT", ""))
    text_encoder_ckpt = args.text_encoder_ckpt or Path(os.environ.get("JOYOMNI_TEXT_ENCODER_CHECKPOINT", ""))

    if not dit_ckpt or not dit_ckpt.exists():
        print("ERROR: DiT checkpoint not found")
        print("Set JOYOMNI_DIT_CHECKPOINT env var or use --dit_ckpt flag")
        sys.exit(1)

    output_path = args.output or f"output_{int(time.time())}.mp4"

    infer(
        prompt=args.prompt,
        input_video=args.input_video,
        output_path=output_path,
        image_path=args.image,
        dit_ckpt=str(dit_ckpt),
        vae_ckpt=str(vae_ckpt),
        text_encoder_ckpt=str(text_encoder_ckpt),
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
