#!/usr/bin/env python3
"""Debug inference issues on horde - detailed output"""

import sys
import os
import torch
import gc
from pathlib import Path

def print_mem(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [{label}] Allocated: {allocated:.1f}GB | Reserved: {reserved:.1f}GB | Total: {total:.1f}GB")

print("=" * 70)
print("HORDE DEBUG - JoyAI Inference (Detailed)")
print("=" * 70)
print()

# 0. System Info
print("[0/6] System Information")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  Working dir: {os.getcwd()}")
print()

# 1. CUDA check
print("[1/6] CUDA Setup")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print_mem("Initial")
print()

# 2. Check model files
print("[2/6] Model Files")
dit_path = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")
vae_path = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")

print(f"  DiT checkpoint: {dit_path}")
if dit_path.exists():
    size = dit_path.stat().st_size / 1e9
    print(f"    ✓ Exists ({size:.2f}GB)")
else:
    print(f"    ✗ NOT FOUND")

print(f"  VAE checkpoint: {vae_path}")
if vae_path.exists():
    files = list(vae_path.glob("*"))
    print(f"    ✓ Exists ({len(files)} files)")
    for f in files[:5]:
        print(f"      - {f.name}")
else:
    print(f"    ✗ NOT FOUND")
print()

# 3. Import test
print("[3/6] Imports")
try:
    sys.path.insert(0, './deploy')
    print("  Importing xvideo modules...")
    from xvideo.models.models import load_dit
    print("    ✓ load_dit")
    from xvideo.models.vae import XVAEChunkCausal
    print("    ✓ XVAEChunkCausal")
    from xvideo.config import ExpConfig
    print("    ✓ ExpConfig")
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 4. Load DiT
print("[4/6] Loading DiT Model")
try:
    device = torch.device("cuda")
    print(f"  Device: {device}")

    print(f"  Creating config...")
    cfg = ExpConfig()
    cfg.training_mode = False
    cfg.dit_precision = "fp16"
    cfg.dit_ckpt = str(dit_path)
    print(f"    Checkpoint: {cfg.dit_ckpt}")
    print(f"    Precision: {cfg.dit_precision}")

    print_mem("Before load_dit")
    print(f"  Loading checkpoint...")
    dit = load_dit(cfg, device=device)
    print(f"    ✓ Loaded from checkpoint")
    print_mem("After load_dit")

    print(f"  Converting to float16...")
    dit = dit.to(device, dtype=torch.float16)
    print(f"    ✓ Model on device in float16")

    print(f"  Converting buffers to float16...")
    for buffer in dit.buffers():
        buffer.data = buffer.data.to(dtype=torch.float16)
    print(f"    ✓ Buffers converted")

    print(f"  Setting eval mode...")
    dit.eval()
    dit.requires_grad_(False)
    print(f"    ✓ Eval mode set")

    print(f"  Clearing memory...")
    gc.collect()
    torch.cuda.empty_cache()
    print_mem("After cleanup")
    print("  ✓ DiT loaded successfully")
except Exception as e:
    print(f"  ✗ DiT load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 5. Load VAE
print("[5/6] Loading VAE Model")
try:
    print_mem("Before VAE load")
    print(f"  Loading from: {vae_path}")
    vae = XVAEChunkCausal.from_pretrained(str(vae_path), torch_dtype=torch.float16)
    print(f"    ✓ Loaded from checkpoint")
    print_mem("After load")

    print(f"  Moving to device...")
    vae = vae.to(device)
    print(f"    ✓ On device")
    print_mem("After to(device)")

    print(f"  Setting eval mode...")
    vae.eval()
    vae.requires_grad_(False)
    print(f"    ✓ Eval mode set")

    print(f"  Clearing memory...")
    gc.collect()
    torch.cuda.empty_cache()
    print_mem("After cleanup")
    print("  ✓ VAE loaded successfully")
except Exception as e:
    print(f"  ✗ VAE load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 6. joyomni_ops check
print("[6/6] joyomni_ops Check")
try:
    from joyomni_ops import fused_norm_scale_shift
    print("  ✓ joyomni_ops available")
except Exception as e:
    print(f"  ⚠ joyomni_ops not available: {e}")
print()

print("=" * 70)
print("✅ DEBUG COMPLETE")
print("=" * 70)
print()
print("Summary:")
print("  - All models loaded successfully in float16")
print(f"  - Final memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
print("  - Ready for inference")
print()
