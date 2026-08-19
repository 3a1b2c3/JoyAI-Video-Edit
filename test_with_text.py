#!/usr/bin/env python3
"""Test with real text embeddings (not random context) using joyomni_ops."""

import os
import sys
from pathlib import Path

os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.getcwd()}/deploy:{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Must import joyomni_ops FIRST, before torch
import joyomni_ops  # noqa: F401

import torch
sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit, load_text_encoder
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything


def test_with_text():
    """Test diffusion with real text embeddings."""
    device = torch.device("cuda")
    seed_everything(42)

    print("=" * 70)
    print("TEST: DiT with real text embeddings")
    print("=" * 70)
    print()

    # Load DiT
    print("[1/4] Loading DiT...")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "bf16"
    cfg.dit_ckpt = str(Path("dit_quantized.pth"))
    dit = load_dit(cfg, device=device)
    dit.eval()
    print(f"  ✓ DiT loaded")
    print()

    # Load text encoder
    print("[2/4] Loading text encoder...")
    text_encoder_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder")
    if not text_encoder_ckpt.exists():
        print(f"  ❌ Text encoder not found: {text_encoder_ckpt}")
        print(f"  Using random context as fallback (but output will be noise)")
        tokenizer, text_encoder = None, None
    else:
        try:
            tokenizer, text_encoder = load_text_encoder(
                str(text_encoder_ckpt),
                device=device,
                torch_dtype=torch.bfloat16
            )
            print(f"  ✓ Text encoder loaded")
        except Exception as e:
            print(f"  ❌ Error loading text encoder: {e}")
            tokenizer, text_encoder = None, None
    print()

    # Create input
    print("[3/4] Creating test inputs...")
    latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.bfloat16, device=device)
    t = torch.tensor([500], device=device, dtype=torch.long)

    # Generate context
    if text_encoder is not None and tokenizer is not None:
        print("  Using real text embeddings:")
        prompt = "A beautiful landscape with mountains and ocean"
        print(f"    Prompt: '{prompt}'")

        # Tokenize
        tokens = tokenizer(prompt, return_tensors="pt")
        print(f"    Tokens shape: {tokens['input_ids'].shape}")

        # Generate embeddings
        with torch.no_grad():
            outputs = text_encoder(**{k: v.to(device) for k, v in tokens.items()})
            context = outputs.hidden_states[-1]  # Use last layer
            print(f"    Context shape: {context.shape}, dtype: {context.dtype}")
            print(f"    Context mean: {context.mean():.6f}, std: {context.std():.6f}")
    else:
        print("  ⚠️  Using random context (fallback - output will be noise)")
        context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)
        print(f"    Context shape: {context.shape}")
    print()

    # Forward pass
    print("[4/4] Running diffusion...")
    with torch.no_grad():
        output = dit(latents, t, context)
        if isinstance(output, tuple):
            output = output[0]

    print(f"  ✓ Output shape: {output.shape}")
    print(f"    mean: {output.mean():.6f}, std: {output.std():.6f}")
    print()

    print("=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print()
    if text_encoder is None:
        print("⚠️  Text encoder not loaded. To get meaningful output:")
        print("  1. Check if text encoder checkpoint exists at:")
        print(f"     {text_encoder_ckpt}")
        print("  2. If not, obtain the checkpoint and place it there")
        print("  3. Then re-run this test")
    else:
        print("✓ Using real text embeddings - output should be meaningful")
    print()


if __name__ == "__main__":
    test_with_text()
