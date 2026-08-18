#!/usr/bin/env python3
"""Debug inference - trace what the model is actually doing."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, './deploy')

from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.utils import seed_everything


def debug_test():
    """Debug test with detailed logging."""
    device = torch.device("cuda")
    seed_everything(42)

    print("=" * 70)
    print("DEBUG TEST: Trace model behavior")
    print("=" * 70)
    print()

    # Load DiT
    print("[1/5] Loading DiT...")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "bf16"
    cfg.dit_ckpt = str(Path("dit_quantized.pth"))
    dit = load_dit(cfg, device=device)
    dit.eval()
    print(f"  ✓ DiT loaded, total params: {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B")
    print()

    # Check model weights
    print("[2/5] Checking model weights...")
    weight_stats = []
    for name, param in list(dit.named_parameters())[:5]:  # First 5 params
        if param.numel() > 0:
            weight_stats.append((name, param.mean().item(), param.std().item(), param.min().item(), param.max().item()))
            print(f"  {name}: mean={param.mean():.6f}, std={param.std():.6f}, min={param.min():.6f}, max={param.max():.6f}")
    print()

    # Create synthetic latents and context
    print("[3/5] Creating test inputs...")
    latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.bfloat16, device=device)
    t = torch.tensor([500], device=device, dtype=torch.long)

    # PROBLEM: Random context is noise. Should be text embeddings.
    context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)

    print(f"  Latents shape: {latents.shape}, dtype: {latents.dtype}")
    print(f"    mean: {latents.mean():.6f}, std: {latents.std():.6f}, min: {latents.min():.6f}, max: {latents.max():.6f}")
    print(f"  Time step: {t.item()}")
    print(f"  Context shape: {context.shape}, dtype: {context.dtype}")
    print(f"    mean: {context.mean():.6f}, std: {context.std():.6f}, min: {context.min():.6f}, max: {context.max():.6f}")
    print(f"  ⚠️  WARNING: Context is random noise, not text embeddings!")
    print()

    # Forward pass
    print("[4/5] Running model forward pass...")
    with torch.no_grad():
        output = dit(latents, t, context)
        if isinstance(output, tuple):
            output = output[0]

    print(f"  ✓ Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"    mean: {output.mean():.6f}, std: {output.std():.6f}, min: {output.min():.6f}, max: {output.max():.6f}")
    print()

    # Check if output changed
    print("[5/5] Checking if model actually processed...")
    diff = (output - latents).abs()
    print(f"  Difference from input: mean={diff.mean():.6f}, std={diff.std():.6f}")
    if diff.mean() < 1e-5:
        print(f"  ❌ ERROR: Output is nearly identical to input! Model didn't process.")
    else:
        print(f"  ✓ Output differs from input (model is processing)")
    print()

    # Check output distribution
    print("Output statistics:")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  Std: {output.std():.6f}")
    print(f"  Min: {output.min():.6f}")
    print(f"  Max: {output.max():.6f}")
    print(f"  Median: {output.median():.6f}")
    print(f"  Num zeros: {(output == 0).sum().item()}")
    print(f"  Num nans: {torch.isnan(output).sum().item()}")
    print(f"  Num infs: {torch.isinf(output).sum().item()}")
    print()

    print("=" * 70)
    print("SOLUTION: Use real text embeddings, not random context")
    print("=" * 70)
    print()
    print("Steps to fix:")
    print("  1. Load text encoder: load_text_encoder(checkpoint)")
    print("  2. Tokenize text: tokenizer(prompt)")
    print("  3. Generate embeddings: text_encoder(tokens)")
    print("  4. Extract context from embeddings")
    print("  5. Use context in diffusion instead of torch.randn()")
    print()


if __name__ == "__main__":
    debug_test()
