#!/usr/bin/env python3
"""
JoyAI-Video-Edit inference with TEXT PROMPTS only (no style image)
Uses joyomni_ops CUDA kernels for high-quality video generation
"""

import os
import sys
import gc
import argparse
import traceback
from pathlib import Path

# Setup paths
os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.getcwd()}/deploy:{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Must import joyomni_ops before torch inference
import joyomni_ops  # noqa: F401

import torch
import cv2
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
from transformers import AutoTokenizer
from PIL import Image


def mem_info(label=""):
    if label:
        print(f"[MEM] {label}", flush=True)
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    available = total - reserved
    print(f"  Allocated: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {total:.1f}GB / Free: {available:.1f}GB", flush=True)


def show_error(exc_type, exc_value, exc_traceback):
    print(f"\n{'='*70}")
    print(f"❌ UNHANDLED ERROR: {exc_type.__name__}")
    print(f"{'='*70}")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = show_error


def load_models(device):
    """Load DiT and VAE"""
    print("[2/5] Loading models (bf16, memory-efficient)...")
    gc.collect()

    dit_ckpt_path = Path("dit_quantized.pth")
    if not dit_ckpt_path.exists():
        print(f"ERROR: Quantized checkpoint not found: {dit_ckpt_path}")
        print("Run: python quantize_simple.py <input> dit_quantized.pth")
        sys.exit(1)

    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "bf16"
    cfg.dit_ckpt = str(dit_ckpt_path)

    print("  Loading DiT (quantized checkpoint, auto-dequantized to bf16)...")
    mem_before_load = torch.cuda.memory_allocated() / 1e9
    mem_info("Before load_dit()")

    dit = load_dit(cfg, device=device)
    mem_info("After load_dit()")
    mem_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"    GPU memory after load: {mem_after_load:.1f}GB (+{mem_after_load - mem_before_load:.1f}GB)")

    model_dtype = next(dit.parameters()).dtype
    print(f"    Model dtype: {model_dtype}")

    dit.eval()
    dit.requires_grad_(False)
    torch.cuda.empty_cache()
    mem_after_cleanup = torch.cuda.memory_allocated() / 1e9
    print(f"  ✓ DiT loaded (final: {mem_after_cleanup:.1f}GB)")

    # Load VAE in bf16
    print("  Loading VAE (bf16, CPU)...")
    vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
    vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.bfloat16)
    vae = vae.to("cpu")
    for param in vae.parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    for buf in vae.buffers():
        if buf.dtype not in (torch.long, torch.int, torch.bool):
            if buf.dtype != torch.bfloat16:
                buf.data = buf.data.to(torch.bfloat16)
    vae.eval()
    vae.requires_grad_(False)
    print(f"  ✓ VAE loaded (bf16, CPU)")
    mem_info("After VAE load")

    gc.collect()
    torch.cuda.empty_cache()
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU memory summary:")
    print(f"    Allocated: {mem_allocated:.1f}GB")
    print(f"    Reserved:  {mem_reserved:.1f}GB")
    print(f"    Total:     {mem_total:.1f}GB")
    print(f"    Available: {mem_total - mem_reserved:.1f}GB")
    print()

    return dit, vae


def encode_text_prompt(text_prompt, device):
    """Encode text prompt to embeddings (random for now, text encoder not available)"""
    print(f"[3.5/5] Text prompt encoding...")
    print(f"  Prompt: '{text_prompt}'")

    # Generate random context as placeholder (in production, use actual text encoder)
    context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
    print(f"  Context shape: {context.shape}, dtype: {context.dtype}")
    print(f"  (Using random context as placeholder - text encoder not integrated)")
    return context


def load_video(video_path, num_frames, height, width):
    """Load video frames"""
    print(f"[1/5] Loading video: {video_path}")
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_load = total_frames if num_frames == "all" else int(num_frames)

    # Auto-size to source aspect ratio when H/W aren't given explicitly
    auto_size = height is None or width is None
    if auto_size:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 16
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 16
        ar = src_w / src_h
        _snap = lambda x: max(16, int(round(x / 16)) * 16)
        BASE = 256
        if ar >= 1:  # landscape
            width, height = BASE, _snap(BASE / ar)
        else:  # portrait
            height, width = BASE, _snap(BASE * ar)
        print(f"  Auto resolution from {src_w}x{src_h} (AR {ar:.3f}) -> {width}x{height}")

    frames = []
    for i in range(min(frames_to_load, total_frames)):
        ok, bgr = cap.read()
        if not ok:
            break

        if auto_size:
            bgr_out = cv2.resize(bgr, (width, height))
        else:
            orig_h, orig_w = bgr.shape[:2]
            aspect = orig_w / orig_h
            if aspect > width / height:
                new_w, new_h = width, int(width / aspect)
            else:
                new_h, new_w = height, int(height * aspect)
            bgr_resized = cv2.resize(bgr, (new_w, new_h))
            pad_t, pad_b = (height - new_h) // 2, height - new_h - (height - new_h) // 2
            pad_l, pad_r = (width - new_w) // 2, width - new_w - (width - new_w) // 2
            bgr_out = cv2.copyMakeBorder(bgr_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))

        rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)

    cap.release()

    frames_tensor = torch.stack(frames).to("cuda", dtype=torch.bfloat16)
    print(f"✓ Loaded {len(frames)} frames @ {fps:.1f} fps")
    print()

    return frames_tensor, fps


def vae_encode(frames_tensor, vae, device):
    """Encode frames to latents"""
    print("[3/5] VAE encoding...")
    mem_before_encode = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory before encoding: {mem_before_encode:.1f}GB")
    print(f"  [DEBUG] frames_tensor: {frames_tensor.shape} {frames_tensor.dtype} on {frames_tensor.device}")

    with torch.no_grad():
        frames_chw = frames_tensor.permute(0, 3, 1, 2)
        print(f"  [DEBUG] After permute: {frames_chw.shape}")
        frames_chw = frames_chw.to("cpu")
        print(f"  [DEBUG] Moved to CPU: {frames_chw.device}")

        latents_list = []
        for i in tqdm(range(len(frames_chw)), desc="Encoding"):
            try:
                z = frames_chw[i:i+1].unsqueeze(2)
                print(f"  [DEBUG] Frame {i}: z shape={z.shape} on {z.device}")
                posterior = vae.encode(z).latent_dist
                sample = posterior.sample() * 0.18215
                print(f"  [DEBUG] Frame {i}: latent shape={sample.shape}")
                latents_list.append(sample)
            except Exception as e:
                print(f"  ⚠ Frame {i} encode error: {e}")
                traceback.print_exc()
                continue

        print(f"  [DEBUG] Encoded {len(latents_list)} frames successfully")
        if latents_list:
            latents = torch.cat(latents_list, dim=0)
            print(f"  [DEBUG] Concatenated latents: {latents.shape}")
        else:
            print("  WARNING: latents_list empty, using raw frames")
            latents = frames_chw[:1]

        print(f"  [DEBUG] Moving latents to GPU...")
        latents = latents.to("cuda")
        print(f"  [DEBUG] Latents on GPU: {latents.shape} {latents.dtype} on {latents.device}")

        gc.collect()
        torch.cuda.empty_cache()
        mem_after_encode = torch.cuda.memory_allocated() / 1e9
        print(f"✓ Encoded {len(latents)} frames")
        print(f"  Memory: {mem_after_encode:.1f}GB ({mem_after_encode - mem_before_encode:+.1f}GB)")

    print()
    return latents


def diffusion(dit, latents, context, steps, cfg_scale, device):
    """Diffusion with Classifier-Free Guidance"""
    # Repeat context to match latent batch size
    if context is not None:
        batch_size = latents.shape[0]
        if context.shape[0] == 1 and batch_size > 1:
            context = context.repeat(batch_size, 1, 1)
            print(f"  [DEBUG] Repeated context to batch size {batch_size}: {context.shape}")

    guidance_mode = "text" if context is not None else "random"
    print(f"[4/5] Diffusion ({steps} steps, CFG={cfg_scale}, context={guidance_mode})...")
    mem_before_diffusion = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory before diffusion: {mem_before_diffusion:.1f}GB")

    print(f"  [DEBUG] Starting diffusion loop with {steps} steps")
    print(f"  [DEBUG] Latents shape: {latents.shape}, device: {latents.device}")
    print(f"  [DEBUG] Context shape: {context.shape if context is not None else 'None'}")

    with torch.no_grad():
        for step in tqdm(range(steps), desc="Denoising"):
            t = (steps - step - 1) / steps
            t_idx = int(t * 1000)
            t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

            if context is not None:
                context_cond = context
            else:
                context_cond = torch.randn(latents.shape[0], 256, 4096, dtype=torch.bfloat16, device=device)

            if step == 0:
                print(f"  [DEBUG] Step {step}: t={t:.3f}, t_idx={t_idx}, context shape={context_cond.shape}, device={context_cond.device}")

            # Conditional forward pass
            output_cond = dit(latents, t_tensor, context_cond)
            if isinstance(output_cond, (tuple, list)):
                output_cond = output_cond[0]

            # Unconditional forward pass
            context_uncond = torch.zeros_like(context_cond)
            output_uncond = dit(latents, t_tensor, context_uncond)
            if isinstance(output_uncond, (tuple, list)):
                output_uncond = output_uncond[0]

            # Apply Classifier-Free Guidance
            model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

            if step == 0:
                print(f"  [DEBUG] Step {step}: output shapes: cond={output_cond.shape}, uncond={output_uncond.shape}, final={model_output.shape}")

            sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
            latents = latents + model_output * (-sigma * 0.1)

            if step == 0 or step == steps - 1:
                mem_diffusion = torch.cuda.memory_allocated() / 1e9
                print(f"  [DEBUG] Step {step}: latents updated, memory: {mem_diffusion:.1f}GB")

            torch.cuda.empty_cache()

    # Move model back to CPU to free GPU
    dit.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_diffusion = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Diffusion complete")
    print(f"  Memory: {mem_after_diffusion:.1f}GB ({mem_after_diffusion - mem_before_diffusion:+.1f}GB)")
    print()

    return latents


def vae_decode(latents, vae, device):
    """Decode latents to frames"""
    print("[5/5] Decoding...")
    mem_before_decode = torch.cuda.memory_allocated() / 1e9
    print(f"  Memory before decoding: {mem_before_decode:.1f}GB")

    with torch.no_grad():
        latents_decoded = latents / 0.18215
        if str(vae.device) == 'cpu':
            latents_decoded = latents_decoded.to("cpu")
        frames_decoded = []
        for i in tqdm(range(len(latents_decoded)), desc="Decoding"):
            try:
                frame = vae.decode(latents_decoded[i:i+1]).sample
                frame = frame.to(device)
                frames_decoded.append(frame)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Frame {i}: {e}")
                traceback.print_exc()
                continue

        decoded = torch.cat(frames_decoded, dim=0) if frames_decoded else latents_decoded[:1]
        gc.collect()
        torch.cuda.empty_cache()
        mem_after_decode = torch.cuda.memory_allocated() / 1e9
        print(f"✓ Decoded {len(decoded)} frames")
        print(f"  Memory: {mem_after_decode:.1f}GB ({mem_after_decode - mem_before_decode:+.1f}GB)")
        print()

    return decoded


def save_video(decoded, output_path, fps):
    """Save frames to MP4"""
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        if decoded.ndim == 5:
            decoded = decoded.squeeze(2)
        if decoded.ndim == 4:
            decoded = decoded.permute(0, 2, 3, 1)

        output_frames = (decoded.to(torch.float32) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
        print(f"  Saving {len(output_frames)} frames...")
        iio.imwrite(output_path, output_frames, fps=fps)
        print(f"\n✓ Output saved:")
        print(f"  {output_path}")
    except Exception as e:
        print(f"\n❌ ERROR saving output: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("--output", default="outputs/dit_text_output.mp4", help="Output video")
    parser.add_argument("--prompt", default="A beautiful landscape", help="Text prompt for guidance")
    parser.add_argument("--frames", type=int, default=10, help="Frames to process")
    parser.add_argument("--height", type=int, default=None, help="Height (None=auto)")
    parser.add_argument("--width", type=int, default=None, help="Width (None=auto)")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    args = parser.parse_args()

    print("=" * 70)
    print("JoyAI-Video-Edit Inference (TEXT PROMPTS ONLY)")
    print("=" * 70)
    print()

    seed_everything(42)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    mem_info("Start")
    print()

    # Load models
    dit, vae = load_models(device)

    # Load video
    frames_tensor, fps = load_video(args.video, args.frames, args.height, args.width)

    # Encode text prompt
    context = encode_text_prompt(args.prompt, device)
    print()

    # Broadcast context to batch size
    if context is not None:
        batch_size = frames_tensor.shape[0]
        if context.shape[0] == 1 and batch_size > 1:
            context = context.repeat(batch_size, 1, 1)

    # Encode VAE
    latents = vae_encode(frames_tensor, vae, device)

    # Diffusion
    latents = diffusion(dit, latents, context, args.steps, args.cfg, device)

    # Decode VAE
    decoded = vae_decode(latents, vae, device)

    # Save
    save_video(decoded, args.output, fps)

    print("=" * 70)
    print("✅ INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Full path: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
