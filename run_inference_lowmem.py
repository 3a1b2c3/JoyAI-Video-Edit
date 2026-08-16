#!/usr/bin/env python3
"""Run DiT inference with memory-efficient loading."""

import sys
import argparse
from pathlib import Path
import time
import gc

import torch
import cv2
import numpy as np
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.models.pipeline import PRECISION_TO_TYPE

def load_dit_lowmem(cfg, device):
    """Load DiT with memory-efficient float16."""
    print("Loading DiT (float16, memory-efficient)...")

    # Set float16 precision in config
    print("  [1] Loading from checkpoint directly to GPU...")
    original_precision = cfg.dit_precision
    cfg.dit_precision = "fp16"

    # Load checkpoint directly to GPU (not CPU) to avoid peak RAM spike
    dit = load_dit(cfg, device=device)
    cfg.dit_precision = original_precision

    # Ensure float16 on all parameters and buffers
    print("  [2] Ensuring float16 on GPU...")
    dit = dit.to(device, dtype=torch.float16)

    # Force all buffers to same dtype
    for buffer in dit.buffers():
        buffer.data = buffer.data.to(dtype=torch.float16)

    dit.eval()
    dit.requires_grad_(False)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    return dit

def main():
    parser = argparse.ArgumentParser(description="DiT inference (low memory)")
    parser.add_argument("--video", default="assets/Recording 2026-08-12 205529.mp4")
    parser.add_argument("--out", default="outputs/dit_output_lowmem.mp4")
    parser.add_argument("--ref-image", default="assets/image.png")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("DiT Inference (Low Memory - Float16)")
    print("=" * 70)

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

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

        # Preserve aspect ratio: resize to fit target size, then pad
        orig_h, orig_w = bgr.shape[:2]
        aspect_ratio = orig_w / orig_h

        # Calculate new size maintaining aspect ratio
        if aspect_ratio > args.width / args.height:
            # Wider than target - fit to width
            new_w = args.width
            new_h = int(args.width / aspect_ratio)
        else:
            # Taller than target - fit to height
            new_h = args.height
            new_w = int(args.height * aspect_ratio)

        # Resize
        bgr_resized = cv2.resize(bgr, (new_w, new_h))

        # Pad to target size (center the image)
        pad_top = (args.height - new_h) // 2
        pad_bottom = args.height - new_h - pad_top
        pad_left = (args.width - new_w) // 2
        pad_right = args.width - new_w - pad_left

        bgr_padded = cv2.copyMakeBorder(
            bgr_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)
    cap.release()

    frames_tensor = torch.stack(frames).to(device, dtype=torch.float16)
    print(f"✓ Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Load reference image (style frame)
    ref_image = None
    if args.ref_image and Path(args.ref_image).exists():
        ref_image = Image.open(args.ref_image).convert("RGB")
        print(f"✓ Loaded reference image: {args.ref_image}")
    elif args.ref_image:
        print(f"⚠ Reference image not found: {args.ref_image}")

    # Load models with memory efficiency
    print(f"\n[2/5] Loading models (memory-efficient)...")
    try:
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")

        dit = load_dit_lowmem(cfg, device)
        print(f"✓ DiT loaded (float16)")

        vae_ckpt = DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/vae"
        vae = XVAEChunkCausal.from_pretrained(
            str(vae_ckpt),
            torch_dtype=torch.float16,
        )
        vae = vae.to(device)
        vae.eval()
        vae.requires_grad_(False)
        print(f"✓ VAE loaded (float16)")

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Encode frames
    print(f"\n[3/5] VAE encoding {len(frames)} frames...")
    with torch.no_grad():
        frames_chw = frames_tensor.permute(0, 3, 1, 2)

        latents_list = []
        for i in tqdm(range(len(frames_chw)), desc="Encoding"):
            try:
                frame_i = frames_chw[i:i+1]
                frame_i = frame_i.unsqueeze(2)

                posterior = vae.encode(frame_i).latent_dist
                z = posterior.sample() * getattr(vae.config, 'scaling_factor', 0.18215)
                latents_list.append(z)

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n  ✗ Frame {i}: {e}")
                continue

        if not latents_list:
            print("ERROR: All frames failed to encode")
            return 1

        latents = torch.cat(latents_list, dim=0)
        print(f"✓ Encoded to latents: {latents.shape}")

    # Diffusion
    print(f"\n[4/5] Running diffusion ({args.steps} steps)...")
    with torch.no_grad():
        start = time.time()
        try:
            for step in tqdm(range(args.steps), desc="Denoising"):
                t = (args.steps - step - 1) / args.steps
                t_idx = int(t * 1000)
                t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

                context = torch.randn(latents.shape[0], 256, 4096, dtype=torch.float16, device=device)

                # Forward pass
                model_output = dit(latents, t_tensor, context)

                if isinstance(model_output, (tuple, list)):
                    model_output = model_output[0]

                # Update latents
                sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
                dt = -sigma * 0.1
                latents = latents + model_output * dt

                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        elapsed = time.time() - start
        print(f"Inference time: {elapsed:.2f}s")

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

                torch.cuda.empty_cache()

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

    # Convert back to uint8
    output_frames = (decoded_frames.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        iio.imwrite(args.out, output_frames, fps=fps)
        out_size = Path(args.out).stat().st_size / 1e6
        print(f"\n✓ Video saved: {args.out}")
        print(f"  Size: {out_size:.1f} MB")
        print(f"  Frames: {len(output_frames)}")
    except Exception as e:
        print(f"ERROR: Failed to save video: {e}")
        return 1

    print("\n" + "=" * 70)
    print("✅ LOW MEMORY INFERENCE COMPLETE")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
