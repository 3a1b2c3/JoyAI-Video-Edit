#!/bin/bash
# Download and organize model checkpoints

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/deploy/deps/checkpoints"

echo "========================================================================"
echo "Checkpoint Setup"
echo "========================================================================"
echo ""

# Create directory structure
echo "[1/3] Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/JoyAI-Video-Edit/dit/dit"
mkdir -p "$CHECKPOINT_DIR/JoyAI-Video-Edit/vae"
echo "  ✅ Directories created"
echo ""

# Check existing checkpoints
echo "[2/3] Checking for existing checkpoints..."
echo ""

DIT_PATH="$CHECKPOINT_DIR/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth"
if [ -f "$DIT_PATH" ]; then
    SIZE=$(du -h "$DIT_PATH" | cut -f1)
    echo "  ✅ DiT checkpoint found ($SIZE)"
else
    echo "  ⚠ DiT checkpoint NOT found"
    echo "    Expected at: $DIT_PATH"
    echo "    Size: ~28 GB"
fi

VAE_PATH="$CHECKPOINT_DIR/JoyAI-Video-Edit/vae"
if [ -f "$VAE_PATH/config.json" ] || [ -f "$VAE_PATH/diffusion_pytorch_model.bin" ]; then
    echo "  ✅ VAE checkpoint found"
else
    echo "  ⚠ VAE checkpoint NOT found"
    echo "    Expected at: $VAE_PATH"
fi
echo ""

# Summary and instructions
echo "[3/3] Setup Summary"
echo "========================================================================"
echo ""

if [ -f "$DIT_PATH" ] && [ -f "$VAE_PATH/config.json" ]; then
    echo "✅ All checkpoints ready!"
    echo ""
    echo "You can now run inference:"
    echo "  bash run_inference.sh"
else
    echo "⚠ Missing checkpoints. Please download:"
    echo ""

    if [ ! -f "$DIT_PATH" ]; then
        echo "1. DiT Checkpoint (28-30 GB)"
        echo "   Name: joyai_video_edit_dit_0804.pth"
        echo "   Path: $DIT_PATH"
        echo "   Download from: [Your model source]"
        echo ""
    fi

    if [ ! -f "$VAE_PATH/config.json" ]; then
        echo "2. VAE Checkpoint"
        echo "   Path: $VAE_PATH"
        echo "   Download from: [Your VAE source]"
        echo ""
    fi

    echo "After downloading, verify with:"
    echo "  bash check_environment.sh"
fi

echo ""
echo "========================================================================"
echo "Directory Structure"
echo "========================================================================"
echo ""
echo "deploy/deps/checkpoints/"
echo "├── JoyAI-Video-Edit/"
echo "│   ├── dit/"
echo "│   │   └── dit/"
echo "│   │       └── joyai_video_edit_dit_0804.pth (28-30 GB)"
echo "│   └── vae/"
echo "│       ├── config.json"
echo "│       └── diffusion_pytorch_model.bin"
echo ""
