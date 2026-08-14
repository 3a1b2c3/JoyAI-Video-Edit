#!/usr/bin/env python3
"""
JoyAI-Video-Edit Headless Inference Example
Demonstrates full pipeline (will fail on DiT load without 32GB weights)
"""

import sys
from pathlib import Path

# Add deploy to path
deploy_dir = Path(__file__).parent / "deploy"
sys.path.insert(0, str(deploy_dir))

import torch
import numpy as np
from PIL import Image

def main():
    print("=" * 70)
    print("JoyAI-Video-Edit Headless Inference Example")
    print("=" * 70)
    print()

    # GPU check
    print("GPU Status:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load text encoder (MiMo-VL)
    print("Loading MiMo-VL text encoder...")
    try:
        from transformers import AutoModel, AutoTokenizer

        mimo_path = deploy_dir / "deps" / "checkpoints" / "MiMo-VL-7B-RL-2508"
        if mimo_path.exists():
            print(f"  ✓ Found: {mimo_path}")
            # tokenizer = AutoTokenizer.from_pretrained(str(mimo_path), trust_remote_code=True)
            # print(f"  ✓ Tokenizer loaded")
        else:
            print(f"  ✗ Not found: {mimo_path}")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    print()

    # Load VAE
    print("Loading VAE (video codec)...")
    try:
        from diffusers import AutoencoderKL
        vae_path = deploy_dir / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "vae"
        if vae_path.exists():
            print(f"  ✓ Found: {vae_path}")
            vae = AutoencoderKL.from_pretrained(str(vae_path), low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
            print(f"  ✓ VAE loaded ({sum(p.numel() for p in vae.parameters()):,} params)")
        else:
            print(f"  ✗ Not found: {vae_path}")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    print()

    # Load DiT (will fail without weights)
    print("Loading DiT (16B diffusion transformer)...")
    try:
        dit_path = deploy_dir / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "dit" / "joyai_video_edit_dit_0811.pth"
        if dit_path.exists():
            print(f"  ✓ Found: {dit_path}")
            print(f"  ✓ Size: {dit_path.stat().st_size / 1e9:.1f} GB")
            # Load weights
            dit_weights = torch.load(dit_path, map_location='cpu')
            print(f"  ✓ DiT weights loaded ({len(dit_weights)} parameter tensors)")
        else:
            print(f"  ✗ Not found: {dit_path}")
            print(f"    Download with: python download.py")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

    # Example pipeline
    print("=" * 70)
    print("Example: Edit video with prompt")
    print("=" * 70)
    print()

    example_prompt = "A person in a futuristic neon outfit, glowing cyberpunk aesthetic"
    print(f"Prompt: {example_prompt}")
    print()

    if not dit_path.exists():
        print("⚠ DiT weights not available (32GB download required)")
        print()
        print("To complete setup:")
        print("  1. Free 50GB disk space")
        print("  2. Run: python download.py")
        print("  3. Re-run this script")
        print()
        return 1

    print("Pipeline steps:")
    print("  1. Encode prompt with MiMo-VL")
    print("  2. Load reference image (identity target)")
    print("  3. Load video frames")
    print("  4. Encode frames with VAE")
    print("  5. Run 8-step DiT diffusion")
    print("  6. Decode with VAE")
    print("  7. Save output MP4")
    print()
    print("✓ Ready to process video")
    return 0

if __name__ == "__main__":
    sys.exit(main())
