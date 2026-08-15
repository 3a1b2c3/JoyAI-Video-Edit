#!/bin/bash
# Complete setup: venv, dependencies, and checkpoint verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_REQUIRED="3.11"  # Force Python 3.11 (NOT 3.12 on Windows)

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    DiT Inference Complete Setup                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Find Python (use what's available)
echo "Finding Python installation..."
PYTHON=""
for py_cmd in python3 python python.exe; do
    if command -v $py_cmd &> /dev/null; then
        PYTHON=$py_cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python not found in PATH"
    exit 1
fi

PYTHON_PATH=$($PYTHON -c "import sys; print(sys.executable)")
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "  Using: $PYTHON_PATH"
echo "  Version: Python $PYTHON_VERSION"

# Warn if Python is old/new, but don't block
MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -ne 3 ]; then
    echo "  ⚠ Python 3 required (you have $MAJOR.x)"
elif [ "$MINOR" -lt 10 ]; then
    echo "  ⚠ Python 3.10+ recommended (you have 3.$MINOR)"
elif [ "$MINOR" -gt 11 ]; then
    echo "  ⚠ Python 3.12+ may have issues on Windows (you have 3.$MINOR)"
else
    echo "  ✅ Python version OK"
fi

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
