#!/usr/bin/env python3
"""JoyAI-Video-Edit inference pipeline - reusable core logic"""

import os
import sys
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
from xvideo.models.pipeline import Pipeline
import imageio.v3 as iio


def mem_info(label=""):
    """Print GPU memory stats"""
    if label:
        print(f"[MEM] {label}")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Allocated: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {total:.1f}GB / Free: {total - reserved:.1f}GB")


def load_models(device):
    """Load DiT, VAE, and Qwen2.5-VL"""
    print("[1/5] Loading models...")
    mem_info("Start")

    # Load DiT
    print("  Loading DiT (quantized, bf16)...")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "bf16"
    cfg.dit_ckpt = str(Path("dit_quantized.pth"))
    dit = load_dit(cfg, device=device)
    dit.eval()
    mem_info("After DiT")

    # Load VAE
    print("  Loading VAE (bf16, CPU)...")
    vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
    vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.bfloat16)
    vae = vae.to("cpu")
    vae.eval()
    mem_info("After VAE")

    # Load Qwen2.5-VL
    print("  Loading Qwen2.5-VL...")
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            local_files_only=False,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        qwen_model.eval()
        print("  ✓ Qwen2.5-VL loaded")
    except Exception as e:
        print(f"  ⚠ Qwen2.5-VL failed: {e}")
        qwen_processor = None
        qwen_model = None

    mem_info("After encoders")
    print()

    return dit, vae, qwen_processor, qwen_model


def load_video(video_path, num_frames, height, width):
    """Load and preprocess video frames"""
    print(f"[2/5] Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {frame_w}x{frame_h} @ {fps:.1f}fps, {total} frames total")

    # Auto resolution
    if height == "auto" or width == "auto":
        aspect = frame_w / frame_h
        if aspect > 1.5:
            height = 256
            width = int(height * aspect)
        else:
            width = 256
            height = int(width / aspect)
        print(f"  Auto resolution: {width}x{height}")
    else:
        height, width = int(height), int(width)

    # Load frames
    frames = []
    for _ in range(min(num_frames, total)):
        ret, bgr = cap.read()
        if not ret:
            break

        # Letterbox
        h, w = bgr.shape[:2]
        aspect = w / h
        if aspect > width / height:
            new_w, new_h = width, int(width / aspect)
        else:
            new_h, new_w = height, int(height * aspect)

        bgr_resized = cv2.resize(bgr, (new_w, new_h))
        pad_t = (height - new_h) // 2
        pad_b = height - new_h - pad_t
        pad_l = (width - new_w) // 2
        pad_r = width - new_w - pad_l
        bgr_padded = cv2.copyMakeBorder(bgr_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)

        rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)

    cap.release()
    tensor = torch.stack(frames).to("cuda", dtype=torch.bfloat16)
    print(f"  ✓ Loaded {len(frames)} frames")
    print()

    return tensor, fps


def encode_image(style_image_path, qwen_processor, qwen_model, vae, device):
    """Encode style image to context embeddings"""
    if qwen_processor is None or qwen_model is None:
        raise RuntimeError("Qwen2.5-VL not available")

    if not Path(style_image_path).exists():
        raise FileNotFoundError(f"Style image not found: {style_image_path}")

    print(f"[3.5/5] Encoding style image...")

    style_img = Image.open(style_image_path).convert("RGB")
    print(f"  ✓ Loaded {style_image_path} ({style_img.size})")

    # Encode directly with image token markers
    prompt_text = "Describe the visual style, colors, atmosphere, and artistic composition of this image.\n<|vision_start|><|image_pad|><|vision_end|>"

    inputs = qwen_processor(text=prompt_text, images=[style_img], return_tensors="pt")

    with torch.no_grad():
        outputs = qwen_model(**{k: v.to("cpu") for k, v in inputs.items()}, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-1]

    # Trim/pad to [1, 256, 4096]
    seq_len = prompt_embeds.shape[1]
    if seq_len >= 256:
        context = prompt_embeds[:, :256, :]
    else:
        pad_size = 256 - seq_len
        context = torch.cat([
            prompt_embeds,
            torch.zeros(1, pad_size, prompt_embeds.shape[-1], device=prompt_embeds.device)
        ], dim=1)

    context = context.to(device).to(torch.bfloat16)
    print(f"  ✓ Context shape: {context.shape}")
    print()

    return context


def encode_vae(frames, vae, device):
    """VAE encode frames to latents"""
    print("[3/5] VAE encoding...")
    mem_info("Before encode")

    with torch.no_grad():
        frames_chw = frames.permute(0, 3, 1, 2).to("cpu")
        latents = []

        for i in tqdm(range(len(frames_chw)), desc="Encoding"):
            z = frames_chw[i:i+1].unsqueeze(2)
            posterior = vae.encode(z).latent_dist
            sample = posterior.sample() * 0.18215
            latents.append(sample.to(device))
            torch.cuda.empty_cache()

        latents = torch.cat(latents, dim=0)

    mem_info("After encode")
    print(f"✓ Encoded {len(latents)} frames")
    print()

    return latents


def diffusion(dit, latents, context, cfg_scale, steps, device):
    """Diffusion with CFG"""
    print(f"[4/5] Diffusion ({steps} steps, CFG={cfg_scale})...")
    mem_info("Before diffusion")

    with torch.no_grad():
        for step in tqdm(range(steps), desc="Denoising"):
            t = (steps - step - 1) / steps
            t_idx = int(t * 1000)
            t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

            # Conditional
            output_cond = dit(latents, t_tensor, context)
            if isinstance(output_cond, (tuple, list)):
                output_cond = output_cond[0]

            # Unconditional
            context_uncond = torch.zeros_like(context)
            output_uncond = dit(latents, t_tensor, context_uncond)
            if isinstance(output_uncond, (tuple, list)):
                output_uncond = output_uncond[0]

            # CFG blend
            model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

            sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
            latents = latents + model_output * (-sigma * 0.1)
            torch.cuda.empty_cache()

    mem_info("After diffusion")
    print()

    return latents


def decode_vae(latents, vae, device):
    """VAE decode latents to frames"""
    print("[5/5] Decoding...")
    mem_info("Before decode")

    with torch.no_grad():
        frames = []
        for i in tqdm(range(len(latents)), desc="Decoding"):
            z = latents[i:i+1].to("cpu") / 0.18215
            frame = vae.decode(z).sample
            frames.append(frame.to(device))
            torch.cuda.empty_cache()

        decoded = torch.cat(frames, dim=0)

    mem_info("After decode")
    print(f"✓ Decoded {len(decoded)} frames")
    print()

    return decoded


def save_video(frames, output_path, fps):
    """Save frames to MP4"""
    print(f"Saving {output_path}...")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim == 5:
        frames = frames.squeeze(2)

    output_frames = (frames.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)

    try:
        iio.imwrite(str(output_path), output_frames, fps=fps)
        assert output_path.exists() and output_path.stat().st_size > 0
        print(f"✓ Saved {output_path} ({output_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise


def run_inference(video_path, output_path, style_image_path, num_frames, height, width, steps, cfg_scale):
    """Full inference pipeline"""
    print("="*70)
    print("JoyAI-Video-Edit Inference Pipeline")
    print("="*70)
    print()

    seed_everything(42)
    device = torch.device("cuda")

    # Load
    dit, vae, qwen_processor, qwen_model = load_models(device)

    # Encode style image
    if style_image_path and Path(style_image_path).exists():
        context = encode_image(style_image_path, qwen_processor, qwen_model, vae, device)
    else:
        print("[3.5/5] Using random context")
        context = None
        print()

    # Load video
    frames, fps = load_video(video_path, num_frames, height, width)

    # Encode VAE
    latents = encode_vae(frames, vae, device)

    # Diffusion
    if context is not None:
        latents = diffusion(dit, latents, context, cfg_scale, steps, device)
    else:
        latents = diffusion(dit, latents, torch.randn_like(latents), cfg_scale, steps, device)

    # Decode VAE
    decoded = decode_vae(latents, vae, device)

    # Save
    save_video(decoded, output_path, fps)

    print()
    print("="*70)
    print("✅ INFERENCE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", default="assets/input.mp4")
    parser.add_argument("--output", default="outputs/dit_output.mp4")
    parser.add_argument("--style", default="assets/image.png")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--height", default="auto")
    parser.add_argument("--width", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=7.5)
    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        output_path=args.output,
        style_image_path=args.style,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        steps=args.steps,
        cfg_scale=args.cfg,
    )
