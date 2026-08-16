#!/usr/bin/env python3
"""Convert JoyAI-Video-Edit DiT to GGUF-like quantized format."""

import sys
from pathlib import Path
import torch
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

def quantize_to_int8(tensor):
    """Quantize tensor to int8."""
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        # Get min/max for quantization
        t_min = tensor.min().item()
        t_max = tensor.max().item()

        # Scale to int8 range (-128, 127)
        scale = (t_max - t_min) / 255.0
        zero_point = int(-t_min / scale) if scale != 0 else 0

        # Quantize
        quantized = torch.round((tensor / scale) + zero_point).clamp(-128, 127).to(torch.int8)

        return {
            'data': quantized,
            'scale': scale,
            'zero_point': zero_point,
            'original_dtype': str(tensor.dtype)
        }
    return {'data': tensor, 'scale': 1.0, 'zero_point': 0}

def convert_dit_to_gguf():
    """Convert DiT to quantized checkpoint."""
    print("=" * 70)
    print("Convert JoyAI-Video-Edit DiT to GGUF (INT8)")
    print("=" * 70)

    device = torch.device("cuda")
    seed_everything(42)

    # Load model
    print("\n[1/3] Loading DiT model...")
    try:
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")

        dit = load_dit(cfg, device=device)
        dit.eval()
        dit.requires_grad_(False)
        print(f"✓ DiT loaded")

        # Count params
        total_params = sum(p.numel() for p in dit.parameters())
        print(f"  Total params: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Quantize all parameters
    print("\n[2/3] Quantizing parameters to INT8...")
    quantized_state = {}
    total_size_original = 0
    total_size_quantized = 0

    for name, param in dit.named_parameters():
        original_size = param.data.numel() * param.data.element_size()
        total_size_original += original_size

        # Quantize
        q_info = quantize_to_int8(param.data)
        quantized_state[name] = q_info

        # Size after quantization (int8 = 1 byte per element + metadata)
        quantized_size = q_info['data'].numel() + 24  # metadata
        total_size_quantized += quantized_size

        if len(quantized_state) % 100 == 0:
            print(f"  Quantized {len(quantized_state)} layers...")

    print(f"✓ Quantized {len(quantized_state)} parameters")
    print(f"  Original size: {total_size_original / 1e9:.2f} GB")
    print(f"  Quantized size: {total_size_quantized / 1e9:.2f} GB")
    print(f"  Compression ratio: {total_size_original / total_size_quantized:.2f}x")

    # Save as GGUF-compatible checkpoint
    print("\n[3/3] Saving GGUF checkpoint...")
    output_path = SCRIPT_DIR / "deploy" / "models" / "dit_int8.gguf.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model': 'joyai_video_edit_dit_0811',
        'format': 'gguf_int8',
        'quantized_state': quantized_state,
        'config': {
            'height': getattr(cfg, 'height', 256),
            'width': getattr(cfg, 'width', 256),
            'num_frames': getattr(cfg, 'num_frames', 16),
            'in_channels': getattr(cfg, 'in_channels', 4),
            'total_params': total_params,
        }
    }, str(output_path), _use_new_zipfile_serialization=False)

    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")

    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nUse with: run_inference_gguf.py")

    return True

if __name__ == "__main__":
    success = convert_dit_to_gguf()
    sys.exit(0 if success else 1)
