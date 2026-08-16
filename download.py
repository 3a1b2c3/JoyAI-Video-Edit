#!/usr/bin/env python3
"""Download JoyAI-Video-Edit checkpoints from HuggingFace."""

import os
import sys
from pathlib import Path

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("ERROR: HF_TOKEN not set")
    print("Set with: set HF_TOKEN=hf_your_token")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

dit_path = Path('deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit')
vae_path = Path('deploy/deps/checkpoints/JoyAI-Video-Edit/vae')

print("Downloading DiT checkpoint (28-30 GB)...")
print(f"  Source: jdopensource/JoyAI-Video-Edit (dit folder)")
print(f"  Destination: {dit_path}")

try:
    dit_file = hf_hub_download(
        repo_id='jdopensource/JoyAI-Video-Edit',
        filename='dit/joyai_video_edit_dit_0811.pth',
        local_dir=str(dit_path),
        token=hf_token,
        resume_download=True
    )
    print(f"  [OK] Downloaded: {dit_file}")
except Exception as e:
    print(f"  ERROR: {e}")
    print()
    print("Check:")
    print("  1. HF_TOKEN is valid and has access to repo")
    print("  2. Repository jdopensource/JoyAI-Video-Edit exists")
    print("  3. File dit/joyai_video_edit_dit_0811.pth exists in repo")
    print("  4. Internet connection is stable")
    print("  5. You have enough disk space (28-30 GB)")
    sys.exit(1)

print()
print("Downloading VAE checkpoint...")
print(f"  Source: xvideo_xvae-released-ckpt")
print(f"  Destination: {vae_path}")

try:
    vae_files = snapshot_download(
        repo_id='xvideo_xvae-released-ckpt',
        local_dir=str(vae_path),
        token=hf_token,
        resume_download=True
    )
    print(f"  [OK] Downloaded to: {vae_path}")
except Exception as e:
    print(f"  ERROR: {e}")
    print()
    print("Check:")
    print("  1. HF_TOKEN is valid")
    print("  2. Repository xvideo_xvae-released-ckpt exists and you have access")
    print("  3. Internet connection is stable")
    print("  4. You have enough disk space")
    sys.exit(1)

print()
print("[OK] All downloads complete!")
