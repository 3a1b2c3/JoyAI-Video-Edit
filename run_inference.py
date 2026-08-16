#!/usr/bin/env python3
"""Run DiT inference with float32 checkpoint."""

import sys
import argparse
from pathlib import Path
import time

import torch
import cv2
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.models.pipeline import PRECISION_TO_TYPE

def print_memory():
    """Print current GPU/CPU memory usage."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU: {gpu_mem:.1f}GB allocated, {gpu_reserved:.1f}GB reserved / 48GB")
    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM: {ram.used / 1e9:.1f}GB used / {ram.total / 1e9:.1f}GB")

def main():
    parser = argparse.ArgumentParser(description="DiT inference (float32)")
    parser.add_argument("--video", default="assets/Recording 2026-08-12 205529.mp4")
    parser.add_argument("--out", default="outputs/dit_output.mp4")
    parser.add_argument("--ref-image", default="assets/image.png", help="Style/reference image (loaded but not used in basic inference)")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("DiT Inference (Float32)")
    print("=" * 70)
    print_memory()

    device = torch.device("cuda")
    seed_everything(args.seed)

    # Load video
    print(f"\n[1/5] Loading video...")
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        return 1

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(min(args.frames, total_frames)):
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (args.width, args.height))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)
    cap.release()

    frames_tensor = torch.stack(frames).to(device)
    print(f"✓ Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Load models
    print(f"\n[2/5] Loading DiT + VAE...")
    try:
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")

        dit = load_dit(cfg, device=device)
        dit = dit.to(device)  # Ensure all params on GPU
        dit.eval()
        dit.requires_grad_(False)
        print(f"✓ DiT loaded (float32)")

        vae_ckpt = DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/vae"
        vae = XVAEChunkCausal.from_pretrained(
            str(vae_ckpt),
            torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision],
        )
        vae = vae.to(device)  # Ensure on GPU
        vae.eval()
        vae.requires_grad_(False)
        print(f"✓ VAE loaded")

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Encode frames
    print(f"\n[3/5] VAE encoding {len(frames)} frames...")
    with torch.no_grad():
        frames_chw = frames_tensor.permute(0, 3, 1, 2)
        vae_dtype = next(vae.parameters()).dtype
        frames_chw = frames_chw.to(dtype=vae_dtype)

        latents_list = []
        for i in tqdm(range(len(frames_chw)), desc="Encoding"):
            try:
                frame_i = frames_chw[i:i+1]
                frame_i = frame_i.unsqueeze(2)
                posterior = vae.encode(frame_i).latent_dist
                z = posterior.sample() * getattr(vae.config, 'scaling_factor', 0.18215)
                latents_list.append(z)
            except Exception as e:
                print(f"\n  ✗ Frame {i}: {e}")
                continue

        if not latents_list:
            print("ERROR: All frames failed to encode")
            return 1

        latents = torch.cat(latents_list, dim=0)
        print(f"✓ Encoded to latents: {latents.shape}")

    # Diffusion with DiT model
    print(f"\n[4/5] Running diffusion ({args.steps} steps)...")
    dit_dtype = next(dit.parameters()).dtype
    with torch.no_grad():
        start = time.time()
        try:
            for step in tqdm(range(args.steps), desc="Denoising"):
                t = (args.steps - step - 1) / args.steps
                t_idx = int(t * 1000)
                t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

                latents_typed = latents.to(dtype=dit_dtype)
                context = torch.randn(latents.shape[0], 256, 4096, dtype=dit_dtype, device=device)

                # Forward pass
                model_output = dit(latents_typed, t_tensor, context)

                # Update latents
                sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
                dt = -sigma * 0.1
                latents = latents + model_output * dt

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ GPU OOM: {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        elapsed = time.time() - start
        print(f"Inference time: {elapsed:.2f}s ({len(frames)/elapsed:.1f} fps)")

    # Decode
    print(f"\n[5/5] Decoding...")
    with torch.no_grad():
        scale = getattr(vae.config, 'scaling_factor', 0.18215)
        latents_decoded = latents / scale

        frames_decoded = []
        for i in tqdm(range(len(latents_decoded)), desc="Decoding"):
            try:
                z = latents_decoded[i:i+1]
                frame = vae.decode(z).sample
                frames_decoded.append(frame)
            except Exception as e:
                print(f"Warning: Frame {i} decode failed ({e})")
                continue

        if not frames_decoded:
            print("ERROR: All frames failed to decode")
            return 1

        decoded_frames = torch.cat(frames_decoded, dim=0)
        print(f"✓ Decoded {len(decoded_frames)} frames")

    # Save
    if decoded_frames.ndim == 5:
        decoded_frames = decoded_frames.squeeze(2)
    output_frames = (decoded_frames.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        iio.imwrite(args.out, output_frames, fps=fps)
        out_size = Path(args.out).stat().st_size / 1e6
        print(f"\n✓ Video saved: {args.out}")
        print(f"  Size: {out_size:.1f} MB")
        print(f"  Frames: {len(output_frames)}")
        print(f"  Duration: {len(output_frames)/fps:.1f}s @ {fps:.1f} fps")
    except Exception as e:
        print(f"ERROR: Failed to save video: {e}")
        return 1

    print("\n" + "=" * 70)
    print("✅ INFERENCE COMPLETE")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
