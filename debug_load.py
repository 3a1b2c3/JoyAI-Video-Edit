#!/usr/bin/env python3
"""Debug checkpoint loading with detailed instrumentation"""

import sys
import torch
from pathlib import Path
import time

sys.path.insert(0, './deploy')

device = torch.device("cuda")
print(f"Device: {device}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print()

# Test 1: Raw torch.load with mmap
print("[1/4] Test torch.load with mmap=True...")
dit_path = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")
print(f"  File: {dit_path}")
print(f"  Size: {dit_path.stat().st_size / 1e9:.1f}GB")

print("  Loading checkpoint (this may take time if mmap'ing)...")
t0 = time.time()
try:
    state_dict = torch.load(dit_path, map_location="cpu", weights_only=True, mmap=True)
    t1 = time.time()
    print(f"  ✓ Loaded in {t1-t0:.1f}s")
    print(f"  State dict keys: {len(state_dict)}")
    print(f"  Sample keys: {list(state_dict.keys())[:3]}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Create model
print()
print("[2/4] Creating model...")
from xvideo.models.models import PRECISION_TO_TYPE, _arch_params
from xvideo.models.dit import Transformer3DModel
from xvideo.config import ExpConfig

cfg = ExpConfig()
cfg.dit_precision = "fp16"

dtype = PRECISION_TO_TYPE[cfg.dit_precision]
print(f"  dtype={dtype}, device={device}")

t0 = time.time()
try:
    model = Transformer3DModel(
        dtype=dtype, device=device, **_arch_params(cfg.dit_arch_config)
    )
    t1 = time.time()
    print(f"  ✓ Model created in {t1-t0:.1f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: Load state dict
print()
print("[3/4] Loading state dict into model...")
print(f"  Model on device: {next(model.parameters()).device}")
print(f"  Model dtype: {next(model.parameters()).dtype}")

# Prepare state dict (same logic as load_dit)
if "model" in state_dict:
    state_dict = state_dict["model"]

for prefix in ("model.", "module.", "transformer."):
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
        break

print(f"  Keys in state_dict: {len(state_dict)}")
print(f"  Sample: {list(state_dict.keys())[:3]}")

# Load state dict with per-tensor conversion
print("  Converting and loading tensors...")
t0 = time.time()
try:
    load_state_dict = {}
    for i, (k, v) in enumerate(state_dict.items()):
        if isinstance(v, torch.Tensor):
            v = v.to(device=device, dtype=dtype, non_blocking=True)
        load_state_dict[k] = v
        if (i + 1) % 100 == 0:
            print(f"    Converted {i+1}/{len(state_dict)} tensors...")

    print(f"  Loading into model...")
    missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=True)
    t1 = time.time()
    print(f"  ✓ Loaded in {t1-t0:.1f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Summary
print()
print("[4/4] Memory summary")
mem_alloc = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  Allocated: {mem_alloc:.1f}GB")
print(f"  Reserved:  {mem_reserved:.1f}GB")
print(f"  Total:     {mem_total:.1f}GB")
print(f"  Free:      {mem_total - mem_reserved:.1f}GB")
print()
print("✅ SUCCESS - Model loaded")
