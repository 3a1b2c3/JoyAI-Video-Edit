#!/bin/bash
# Download model checkpoints from HuggingFace (REQUIRED)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/deploy/deps/checkpoints"

echo "========================================================================"
echo "Download Model Checkpoints"
echo "========================================================================"
echo ""

# MUST have HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable not set"
    echo ""
    echo "Set your HuggingFace token:"
    echo "  export HF_TOKEN=hf_your_actual_token_here"
    echo ""
    echo "Then run:"
    echo "  bash download_checkpoints.sh"
    exit 1
fi

echo "✅ HF_TOKEN is set"
echo ""

# Find Python
PYTHON=""
for py_cmd in python3 python python.exe; do
    if command -v $py_cmd &> /dev/null; then
        PYTHON=$py_cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Create directories
echo "[1/2] Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/JoyAI-Video-Edit/dit/dit"
mkdir -p "$CHECKPOINT_DIR/JoyAI-Video-Edit/vae"
echo "  ✅ Directories created"
echo ""

# Download checkpoints using Python + huggingface_hub
echo "[2/2] Downloading checkpoints..."
echo ""

$PYTHON << 'PYTHON_DOWNLOAD'
import os
import sys
from pathlib import Path

hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    print("❌ ERROR: HF_TOKEN not set")
    sys.exit(1)

# Import huggingface_hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("❌ ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)

dit_path = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit")
vae_path = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")

# Download DiT checkpoint
print("Downloading DiT checkpoint (28-30 GB)...")
print("  Source: jdopensource/JoyAI-Video-Edit (dit folder)")
print("  Destination: " + str(dit_path))

try:
    dit_file = hf_hub_download(
        repo_id="jdopensource/JoyAI-Video-Edit",
        filename="dit/joyai_video_edit_dit_0811.pth",
        local_dir=str(dit_path),
        token=hf_token,
        resume_download=True
    )
    print(f"  ✅ Downloaded: {dit_file}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print("")
    print("Check:")
    print("  1. HF_TOKEN is valid and has access to repo")
    print("  2. Repository jdopensource/JoyAI-Video-Edit exists")
    print("  3. File dit/joyai_video_edit_dit_0811.pth exists in repo")
    print("  4. Internet connection is stable")
    print("  5. You have enough disk space (28-30 GB)")
    sys.exit(1)

# Download VAE checkpoint
print("")
print("Downloading VAE checkpoint...")
print("  Source: xvideo_xvae-released-ckpt")
print("  Destination: " + str(vae_path))

try:
    vae_files = snapshot_download(
        repo_id="xvideo_xvae-released-ckpt",
        local_dir=str(vae_path),
        token=hf_token,
        resume_download=True
    )
    print(f"  ✅ Downloaded to: {vae_path}")

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print("")
    print("Check:")
    print("  1. HF_TOKEN is valid")
    print("  2. Repository xvideo_xvae-released-ckpt exists and you have access")
    print("  3. Internet connection is stable")
    print("  4. You have enough disk space")
    sys.exit(1)

print("")
print("✅ All downloads complete!")

PYTHON_DOWNLOAD

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Download failed. See errors above."
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ Checkpoints Downloaded"
echo "========================================================================"
echo ""
echo "Checkpoint locations:"
echo "  DiT:  $CHECKPOINT_DIR/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth"
echo "  VAE:  $CHECKPOINT_DIR/JoyAI-Video-Edit/vae/"
echo ""
echo "Next: Run inference"
echo "  bash run_inference.sh input.mp4 output.mp4"
echo ""
