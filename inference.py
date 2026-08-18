#!/usr/bin/env python3
"""JoyAI-Video-Edit inference module - core functions for DiT diffusion."""

import sys
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit, load_text_encoder
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything
import imageio.v3 as iio


def show_mem(label: str = ""):
    """Print GPU memory stats."""
    if label:
        print(f"[MEM] {label}")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Allocated: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {total:.1f}GB / Free: {total - reserved:.1f}GB")


def force_dtype(model, dtype: torch.dtype):
    """Force all parameters/buffers to target dtype (fixes bias mismatches)."""
    for param in model.parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)
    for buf in model.buffers():
        if buf.dtype not in (torch.long, torch.int, torch.bool):
            if buf.dtype != dtype:
                buf.data = buf.data.to(dtype)


def generate_context(prompt: str, tokenizer, text_encoder, device: torch.device) -> torch.Tensor:
    """Generate context embeddings from text prompt."""
    print(f"  Prompt: '{prompt}'")

    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    print(f"  Tokens shape: {tokens['input_ids'].shape}")

    # Generate embeddings
    with torch.no_grad():
        outputs = text_encoder(**{k: v.to(device) for k, v in tokens.items()})
        # Extract last layer hidden states as context
        context = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        context = context.to(dtype=torch.bfloat16)  # Ensure bf16

    print(f"  Context shape: {context.shape}, dtype: {context.dtype}")
    print(f"  Context stats: mean={context.mean():.6f}, std={context.std():.6f}")
    return context


def load_models(device: torch.device, precision: str = "bf16"):
    """Load DiT and VAE models."""
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    print("[1/3] Loading DiT...")
    show_mem("Before DiT")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = precision
    cfg.dit_ckpt = str(Path("dit_quantized.pth"))
    dit = load_dit(cfg, device=device)
    dit.eval()
    show_mem("After DiT")

    print("[2/3] Loading VAE...")
    show_mem("Before VAE")
    vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
    vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=dtype)

    # Check available GPU memory - if low, keep VAE on CPU
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
    free_vram = (total_vram - torch.cuda.memory_reserved(device) / 1e9)

    if free_vram > 3.0 and 'cuda' in str(device):
        print(f"  Free VRAM: {free_vram:.1f}GB - loading VAE to GPU")
        vae = vae.to(device)
        vae_device = device
    else:
        print(f"  Free VRAM: {free_vram:.1f}GB - keeping VAE on CPU (low memory)")
        vae = vae.to("cpu")
        vae_device = torch.device("cpu")

    force_dtype(vae, dtype)
    vae.eval()
    show_mem("After VAE")

    return dit, vae, dtype, vae_device


def load_video(video_path: str, num_frames: int, height: int, width: int) -> tuple:
    """Load and preprocess video frames."""
    print(f"\n[3/3] Loading video: {video_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for i in range(min(num_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ok, bgr = cap.read()
        if not ok:
            break

        # Letterbox with aspect ratio preservation
        orig_h, orig_w = bgr.shape[:2]
        aspect = orig_w / orig_h
        if aspect > width / height:
            new_w, new_h = width, int(width / aspect)
        else:
            new_h, new_w = height, int(height * aspect)

        bgr_resized = cv2.resize(bgr, (new_w, new_h))
        pad_t, pad_b = (height - new_h) // 2, height - new_h - (height - new_h) // 2
        pad_l, pad_r = (width - new_w) // 2, width - new_w - (width - new_w) // 2
        bgr_padded = cv2.copyMakeBorder(bgr_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float() / 255.0)

    cap.release()
    frames_tensor = torch.stack(frames).to(torch.device("cuda"), dtype=torch.bfloat16)
    print(f"  ✓ Loaded {len(frames)} frames @ {fps:.1f} fps, shape: {frames_tensor.shape}")
    return frames_tensor, fps


def encode_vae(frames: torch.Tensor, vae, device: torch.device, vae_device: torch.device = None) -> torch.Tensor:
    """Encode video frames to latents using VAE."""
    if vae_device is None:
        vae_device = device
    print("\nEncoding to latents...")
    show_mem("Before VAE encode")

    with torch.no_grad():
        frames_chw = frames.permute(0, 3, 1, 2).to(vae_device)  # Move to VAE device
        latents_list = []

        for i in tqdm(range(len(frames_chw)), desc="Encoding"):
            z = frames_chw[i:i+1].unsqueeze(2)
            posterior = vae.encode(z).latent_dist
            sample = posterior.sample() * 0.18215
            latents_list.append(sample.to(device))  # Move latents back to compute device
            torch.cuda.empty_cache()

        if latents_list:
            latents = torch.cat(latents_list, dim=0)
        else:
            raise RuntimeError("No latents generated")

    show_mem("After VAE encode")
    print(f"  ✓ Encoded to {latents.shape}")
    return latents


def diffusion_step(dit, latents: torch.Tensor, device: torch.device, context: torch.Tensor = None, steps: int = 1):
    """Run diffusion forward pass.

    Args:
        dit: DiT model
        latents: Input latents to refine
        device: Compute device
        context: Context embeddings (text). If None, use random (produces noise!)
        steps: Number of diffusion steps
    """
    print(f"\nDiffusion ({steps} steps)...")
    show_mem("Before diffusion")

    # If no context provided, fall back to random (but warn user)
    if context is None:
        print("  ⚠️  WARNING: No context provided, using random (output will be noise!)")
        context = torch.randn(latents.shape[0], 256, 4096, dtype=torch.bfloat16, device=device)
    else:
        # Ensure context is on device and correct dtype
        context = context.to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for step in tqdm(range(steps), desc="Denoising"):
            t = (steps - step - 1) / steps
            t_idx = int(t * 1000)
            t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

            model_output = dit(latents, t_tensor, context)
            if isinstance(model_output, (tuple, list)):
                model_output = model_output[0]

            sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
            latents = latents + model_output * (-sigma * 0.1)
            torch.cuda.empty_cache()

    show_mem("After diffusion")
    return latents


def decode_vae(latents: torch.Tensor, vae, device: torch.device, vae_device: torch.device = None) -> torch.Tensor:
    """Decode latents to frames using VAE."""
    if vae_device is None:
        vae_device = device
    print("\nDecoding frames...")
    show_mem("Before VAE decode")

    with torch.no_grad():
        latents_decoded = latents / 0.18215
        frames_decoded = []

        for i in tqdm(range(len(latents_decoded)), desc="Decoding"):
            z = latents_decoded[i:i+1].to(vae_device)  # Move to VAE device
            frame = vae.decode(z).sample
            frames_decoded.append(frame.to(device))  # Move back to compute device
            torch.cuda.empty_cache()

        if frames_decoded:
            decoded = torch.cat(frames_decoded, dim=0)
        else:
            raise RuntimeError("No frames decoded")

    show_mem("After VAE decode")
    print(f"  ✓ Decoded {len(decoded)} frames")
    return decoded


def save_video(frames: torch.Tensor, output_path: str, fps: float):
    """Save frames to video file."""
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames.ndim == 5:
        frames = frames.squeeze(2)

    output_frames = (frames.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)

    try:
        iio.imwrite(str(output_path), output_frames, fps=fps)

        # Verify file exists
        assert output_path.exists(), f"File NOT created: {output_path}"
        file_size = output_path.stat().st_size
        assert file_size > 0, f"File empty (0 bytes): {output_path}"

        print(f"  ✓ Saved: {output_path} ({file_size} bytes)")
        return output_path
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def quick_test(device: torch.device = None, prompt: str = None):
    """Quick test with synthetic latents (skip 18min VAE encode).

    Args:
        device: Compute device
        prompt: Text prompt for context. If None, uses random (produces noise)
    """
    if device is None:
        device = torch.device("cuda")

    print("=" * 70)
    print("QUICK TEST: DiT diffusion (synthetic latents)")
    print("=" * 70)
    print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    seed_everything(42)
    show_mem("Start")

    # Load models
    dit, vae, dtype, vae_device = load_models(device)
    print()

    # Load text encoder if prompt provided
    context = None
    tokenizer = None
    text_encoder = None
    if prompt:
        print("[2a/4] Loading text encoder...")
        text_encoder_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder")
        if text_encoder_ckpt.exists():
            try:
                tokenizer, text_encoder = load_text_encoder(
                    str(text_encoder_ckpt),
                    device=device,
                    torch_dtype=dtype
                )
                print(f"  ✓ Text encoder loaded")
                print()

                # Generate context from prompt
                print(f"[2b/4] Generating context from prompt...")
                context = generate_context(prompt, tokenizer, text_encoder, device)
                print()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print(f"  Falling back to random context (output will be noise)")
                print()
        else:
            print(f"  ⚠️  Text encoder not found: {text_encoder_ckpt}")
            print(f"  Falling back to random context (output will be noise)")
            print()

    # Create synthetic latents (skip VAE encoding)
    print("[3/4] Creating synthetic latents...")
    show_mem("Before latents create")
    latents = torch.randn(1, 64, 1, 32, 32, dtype=dtype, device=device)
    print(f"  Latents: {latents.shape}")
    show_mem("After latents create")
    print()

    # Diffusion with context
    latents = diffusion_step(dit, latents, device, context=context, steps=1)
    print()

    # Decode
    print("[4/4] Decoding and saving...")
    show_mem("Before decode")
    decoded = decode_vae(latents, vae, device, vae_device=vae_device)

    # Save
    output_path = save_video(decoded, "outputs/test_quick_output.mp4", fps=24)

    print()
    print("=" * 70)
    print("✅ QUICK TEST COMPLETE")
    if prompt:
        print(f"Prompt: '{prompt}'")
    else:
        print("⚠️  No prompt - output is random (provide --prompt to use real context)")
    print("=" * 70)
    return output_path


def full_inference(video_path: str, output_path: str, prompt: str = None, num_frames: int = 1, height: int = 256, width: int = 256, steps: int = 1, device: torch.device = None):
    """Full inference pipeline: load video -> encode -> diffusion -> decode -> save.

    Args:
        video_path: Path to input video
        output_path: Path to save output video
        prompt: Text prompt for context (IMPORTANT: without this, output is noise!)
        num_frames: Number of frames to process
        height, width: Output resolution
        steps: Number of diffusion steps
        device: Compute device
    """
    if device is None:
        device = torch.device("cuda")

    print("=" * 70)
    print("FULL INFERENCE: Video diffusion pipeline")
    print("=" * 70)
    print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")

    seed_everything(42)
    show_mem("Start")

    # Load models
    dit, vae, dtype, vae_device = load_models(device)
    print()

    # Load text encoder if prompt provided
    context = None
    if prompt:
        print("Loading text encoder...")
        text_encoder_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder")
        if text_encoder_ckpt.exists():
            try:
                tokenizer, text_encoder = load_text_encoder(
                    str(text_encoder_ckpt),
                    device=device,
                    torch_dtype=dtype
                )
                context = generate_context(prompt, tokenizer, text_encoder, device)
                print()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print(f"  Falling back to random context (output will be noise)")
                print()
        else:
            print(f"  ⚠️  Text encoder not found, using random context (output will be noise)")
            print()

    # Load video
    frames, fps = load_video(video_path, num_frames, height, width)
    show_mem("After video load")
    print()

    # Encode
    latents = encode_vae(frames, vae, device, vae_device=vae_device)
    print()

    # Diffusion
    latents = diffusion_step(dit, latents, device, context=context, steps=steps)
    print()

    # Decode
    decoded = decode_vae(latents, vae, device, vae_device=vae_device)
    print()

    # Save
    output_file = save_video(decoded, output_path, fps)

    print()
    print("=" * 70)
    print("✅ INFERENCE COMPLETE")
    if prompt:
        print(f"Prompt: '{prompt}'")
    else:
        print("⚠️  No prompt used - output may be noise")
    print("=" * 70)
    return output_file


if __name__ == "__main__":
    # Default: quick test
    quick_test()
