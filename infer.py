#!/usr/bin/env python3
"""
JoyAI-Video-Edit inference wrapper
Uses joyomni_ops CUDA kernels for high-quality video generation
"""

import os
import sys
import gc
import argparse
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
import traceback as tb_module

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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
    tb_module.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = show_error


def load_models(device):
    """Load DiT, VAE, and Qwen2.5-VL"""
    print("[2/5] Loading models (float16, memory-efficient)...")
    gc.collect()

    # Verify quantized checkpoint exists
    dit_ckpt_path = Path("dit_quantized.pth")
    if not dit_ckpt_path.exists():
        print(f"ERROR: Quantized checkpoint not found: {dit_ckpt_path}")
        print("Run: python quantize_simple.py <input> dit_quantized.pth")
        sys.exit(1)

    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "bf16"
    cfg.dit_ckpt = str(dit_ckpt_path)

    # Load DiT with quantized checkpoint
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

    # Load Qwen2.5-VL
    qwen_processor = None
    qwen_model = None
    try:
        print("  Loading Qwen2.5-VL for image encoding...")
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            local_files_only=False,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        qwen_model.eval()
        print(f"  ✓ Qwen2.5-VL loaded (CPU)")
    except Exception as e:
        print(f"  ⚠ Qwen encoder not available: {e}")

    print(f"  Note: DiT 32GB on GPU + VAE 3GB on CPU + Qwen2.5-VL on CPU")

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

    return dit, vae, qwen_processor, qwen_model


def encode_image_context(style_path, processor, text_encoder, device):
    """Encode style image to context embeddings"""
    if processor is None or text_encoder is None:
        return None

    try:
        img = Image.open(style_path).convert("RGB")
        prompt = "Describe the visual style, colors, atmosphere, and artistic composition.\n<|vision_start|><|image_pad|><|vision_end|>"

        inputs = processor(text=prompt, images=[img], return_tensors="pt")
        with torch.no_grad():
            outputs = text_encoder(**{k: v.to("cpu") for k, v in inputs.items()}, output_hidden_states=True)
            context = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

        # Project to 4096 if needed
        if context.shape[-1] != 4096:
            proj = torch.nn.Linear(context.shape[-1], 4096, dtype=context.dtype, device="cpu")
            with torch.no_grad():
                context = proj(context)

        # Trim/pad to [1, 256, 4096]
        if context.shape[1] >= 256:
            context = context[:, :256, :]
        else:
            pad = torch.zeros(1, 256 - context.shape[1], 4096, dtype=context.dtype)
            context = torch.cat([context, pad], dim=1)

        return context.to(device).to(torch.bfloat16)
    except Exception as e:
        print(f"  ⚠ Image encoding failed: {e}")
        return None


def load_video(path, num_frames, height, width):
    """Load video frames"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for _ in range(num_frames):
        ret, bgr = cap.read()
        if not ret:
            break

        # Letterbox
        h, w = bgr.shape[:2]
        aspect = w / h
        if aspect > width / height:
            nw, nh = width, int(width / aspect)
        else:
            nh, nw = height, int(height * aspect)

        bgr = cv2.resize(bgr, (nw, nh))
        pt, pb = (height - nh) // 2, height - nh - (height - nh) // 2
        pl, pr = (width - nw) // 2, width - nw - (width - nw) // 2
        bgr = cv2.copyMakeBorder(bgr, pt, pb, pl, pr, cv2.BORDER_CONSTANT)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)

    cap.release()
    return torch.stack(frames).to("cuda", dtype=torch.bfloat16), fps


def vae_encode(frames, vae):
    """Encode frames to latents"""
    with torch.no_grad():
        frames_chw = frames.permute(0, 3, 1, 2).to("cpu")
        latents = []

        for i in tqdm(range(len(frames_chw)), desc="VAE encode"):
            z = frames_chw[i:i+1].unsqueeze(2)
            posterior = vae.encode(z).latent_dist
            sample = posterior.sample() * 0.18215
            latents.append(sample.to("cuda"))
            torch.cuda.empty_cache()

        return torch.cat(latents, dim=0)


def diffusion(dit, latents, context, steps, cfg_scale, device):
    """Diffusion with CFG"""
    # Broadcast context to batch size
    if context is not None and context.shape[0] == 1 and latents.shape[0] > 1:
        context = context.repeat(latents.shape[0], 1, 1)

    with torch.no_grad():
        for step in tqdm(range(steps), desc="Diffusion"):
            t = (steps - step - 1) / steps
            t_idx = int(t * 1000)
            t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

            # Conditional
            out_cond = dit(latents, t_tensor, context)
            if isinstance(out_cond, (tuple, list)):
                out_cond = out_cond[0]

            # Unconditional
            ctx_uncond = torch.zeros_like(context) if context is not None else None
            out_uncond = dit(latents, t_tensor, ctx_uncond)
            if isinstance(out_uncond, (tuple, list)):
                out_uncond = out_uncond[0]

            # CFG blend
            if context is not None:
                output = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                output = out_cond

            sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
            latents = latents + output * (-sigma * 0.1)
            torch.cuda.empty_cache()

    return latents


def vae_decode(latents, vae):
    """Decode latents to frames"""
    with torch.no_grad():
        frames = []
        for i in tqdm(range(len(latents)), desc="VAE decode"):
            z = latents[i:i+1].to("cpu") / 0.18215
            frame = vae.decode(z).sample
            frames.append(frame.to("cuda"))
            torch.cuda.empty_cache()

        return torch.cat(frames, dim=0)


def save_video(frames, path, fps):
    """Save frames to MP4"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim == 5:
        frames = frames.squeeze(2)

    output = (frames.float().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
    iio.imwrite(path, output, fps=fps)
    print(f"✓ Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("--output", default="output.mp4", help="Output video")
    parser.add_argument("--style", default="assets/image.png", help="Style image")
    parser.add_argument("--frames", type=int, default=10, help="Frames to process")
    parser.add_argument("--height", type=int, default=256, help="Height")
    parser.add_argument("--width", type=int, default=256, help="Width")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda")

    print("=" * 60)
    print("JoyAI-Video-Edit Inference")
    print("=" * 60)
    print()

    # Load models
    dit, vae, processor, text_encoder = load_models(device)
    print()

    # Encode style
    if Path(args.style).exists():
        print(f"[Style] Encoding {args.style}...")
        context = encode_image_context(args.style, processor, text_encoder, device)
        print()
    else:
        print("[Style] Not found, using random")
        context = None
        print()

    # Load video
    print(f"[Video] Loading {args.video}...")
    frames, fps = load_video(args.video, args.frames, args.height, args.width)
    print(f"  Loaded {len(frames)} frames @ {fps:.1f} fps")
    print()

    # Encode VAE
    print("[VAE] Encoding...")
    latents = vae_encode(frames, vae)
    print()

    # Diffusion
    print(f"[Diffusion] {args.steps} steps, CFG={args.cfg}...")
    latents = diffusion(dit, latents, context, args.steps, args.cfg, device)
    print()

    # Decode VAE
    print("[VAE] Decoding...")
    decoded = vae_decode(latents, vae)
    print()

    # Save
    save_video(decoded, args.output, fps)
    print()
    print("=" * 60)
    print("✅ Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
