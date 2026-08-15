#!/bin/bash
# Complete setup: venv, dependencies, and checkpoint verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    DiT Inference Complete Setup                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Setup venv
echo "Step 1: Setting up virtual environment..."
bash setup_venv.sh
echo ""

# Step 2: Check environment
echo "Step 2: Checking environment..."
bash check_environment.sh
echo ""

# Step 3: Checkpoint setup
echo "Step 3: Verifying checkpoints..."
bash download_checkpoints.sh
echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                         Setup Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "Next: Run inference with one of the following:"
echo ""
echo "  1. Quick start (defaults):"
echo "     bash run_inference.sh"
echo ""
echo "  2. Custom parameters:"
echo "     bash run_inference_custom.sh input.mp4 output.mp4 4 256 256 8 42"
echo ""
echo "  3. Batch processing:"
echo "     bash batch_inference.sh ./videos outputs 4 4"
echo ""
echo "Documentation:"
echo "  - SHELL_SCRIPTS.md — Script usage guide"
echo "  - RUN_INFERENCE.md — Quick reference"
echo "  - QUANTIZATION_ATTEMPTS.md — Technical details"
echo ""
