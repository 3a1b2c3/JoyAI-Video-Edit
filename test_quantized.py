#!/usr/bin/env python3
"""Test loading quantized checkpoint"""

import torch
from pathlib import Path

print("Testing quantized checkpoint loading...")
ckpt_path = "dit_quantized.pth"

if not Path(ckpt_path).exists():
    print(f"❌ Checkpoint not found: {ckpt_path}")
    print("Run: python quantize_simple.py <input> dit_quantized.pth")
    exit(1)

print(f"Loading: {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location="cpu")

print(f"Keys loaded: {len(state_dict)}")
print("Sample keys:")
for k in list(state_dict.keys())[:5]:
    v = state_dict[k]
    if isinstance(v, dict):
        print(f"  {k}: quantized int8, scale={v.get('scale', '?')}")
    elif isinstance(v, torch.Tensor):
        print(f"  {k}: {v.dtype} {v.shape}")
    else:
        print(f"  {k}: {type(v)}")

print("✅ Quantized checkpoint loads OK")
