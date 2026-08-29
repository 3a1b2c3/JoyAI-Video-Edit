#!/bin/bash
# Setup JoyAI-Echo 1.5 FP4 on WSL
set -uo pipefail

echo "🚀 JoyAI-Echo 1.5 FP4 Setup (WSL)"
echo "===================================="
echo

# Verify WSL environment
if ! command -v wsl.exe &> /dev/null && [ ! -f /proc/version ] || ! grep -q microsoft /proc/version; then
  echo "⚠️  Not running in WSL. This script is optimized for WSL2."
fi

# Set cache directory (WSL native for speed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME/hub"
echo "📁 Cache: $HF_HOME/hub"
echo

# Check disk space
AVAILABLE=$(df "$HF_HOME" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))
echo "💾 Available disk space: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 25 ]; then
  echo "❌ ERROR: Need at least 25 GB free (FP4 is ~22.81 GB)"
  exit 1
fi
echo

# Check Python & HF CLI
echo "🔍 Checking dependencies..."
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 not found. Install with: sudo apt update && sudo apt install python3 python3-pip"
  exit 1
fi

# Create venv for WSL (PEP 668 compliance)
VENV_DIR="$HF_HOME/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating Python venv..."
  python3 -m venv "$VENV_DIR" > /dev/null 2>&1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install in venv
echo "📦 Installing huggingface-hub in venv..."
pip install huggingface-hub > /dev/null 2>&1
echo "✓ Dependencies ready (venv: $VENV_DIR)"
echo

# Download FP4 checkpoint only
echo "⏳ Downloading JoyAI-Echo 1.5 FP4 (22.81 GB)..."
echo "   This may take 20-40 minutes depending on connection speed"
echo

python3 << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f"Cache directory: {cache_dir}/hub")

try:
    snapshot_download(
        repo_id="jdopensource/JoyAI-Echo",
        repo_type="model",
        cache_dir=f"{cache_dir}/hub",
        allow_patterns=["echo15_fp4*"],
        resume_download=True
    )
    print("✅ Download complete!")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
  echo "✅ Download complete!"
else
  echo "❌ Download failed. Check internet connection."
  exit 1
fi
echo

# Verify download
ECHO_CACHE="$HF_HOME/hub/models--jdopensource--JoyAI-Echo"
if [ -d "$ECHO_CACHE" ]; then
  SIZE=$(du -sh "$ECHO_CACHE" | cut -f1)
  echo "📊 JoyAI-Echo FP4 installed: $SIZE"
  echo "   Location: $ECHO_CACHE"
else
  echo "⚠️  Cache directory not found. Download may have failed."
  exit 1
fi
echo

# Show ready status
echo "🎉 Setup complete!"
echo
echo "Next steps:"
echo "  1. Mount HF cache in JoyAI-Video-Edit:"
echo "     export HF_HOME=$HF_HOME"
echo
echo "  2. Run server with Echo FP4:"
echo "     export JOYOMNI_MODEL=echo_fp4"
echo "     bash run_server_best.sh"
echo
echo "  3. Access web UI:"
echo "     http://localhost:8000"
echo
