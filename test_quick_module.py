#!/usr/bin/env python3
"""Quick test runner - synthetic latents, skip VAE encoding."""

import os
import sys
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONPATH'] = f"{os.getcwd()}/deploy/joyomni_ops:{os.environ.get('PYTHONPATH', '')}"

from inference import quick_test

if __name__ == "__main__":
    device = torch.device("cuda")

    # Get prompt from command line or use None (random)
    prompt = sys.argv[1] if len(sys.argv) > 1 else None

    if prompt:
        print(f"Using prompt: '{prompt}'")
    else:
        print("No prompt provided - using random context (output will be noise)")
        print("Usage: python test_quick_module.py 'your prompt here'")
    print()

    quick_test(device=device, prompt=prompt)
