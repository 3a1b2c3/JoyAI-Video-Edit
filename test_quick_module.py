#!/usr/bin/env python3
"""Quick test runner - synthetic latents, skip VAE encoding (uses joyomni_ops)."""

import os
import sys

os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.getcwd()}/deploy:{os.environ.get('PYTHONPATH', '')}"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Must import joyomni_ops FIRST, before torch
import joyomni_ops  # noqa: F401

import torch
sys.path.insert(0, './deploy')

# Import from inference_pipeline module
from inference_pipeline import load_models, diffusion

if __name__ == "__main__":
    device = torch.device("cuda")

    print("Quick Test: DiT diffusion with synthetic latents (VAE encoding skipped)")
    print(f"Device: {device}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()

    # Load models
    print("[1/2] Loading models...")
    dit, vae, _, _ = load_models(device)
    print()

    # Test diffusion with synthetic latents
    print("[2/2] Testing diffusion with synthetic latents...")
    latents = torch.randn(1, 64, 1, 32, 32, dtype=torch.bfloat16, device=device)
    context = torch.randn(1, 256, 4096, dtype=torch.bfloat16, device=device)

    print(f"  Latents: {latents.shape}")
    print(f"  Context: {context.shape}")

    latents = diffusion(dit, latents, context, steps=5, cfg_scale=7.5, device=device)
    print(f"  ✓ Diffusion complete: {latents.shape}")
    print()
    print("✅ Quick test passed!")
