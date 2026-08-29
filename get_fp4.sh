#!/bin/bash
# Download JoyAI-Echo FP4 model to WSL cache
set -uo pipefail

echo "Downloading JoyAI-Echo FP4 (22.81 GB)..."
echo "This may take 30-60 minutes depending on connection speed"
echo

# Set WSL cache directory
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME/hub"
echo "Cache: $HF_HOME/hub"
echo

# Check disk space
AVAILABLE=$(df "$HF_HOME" | awk 'NR==2 {print $4}')
AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))
echo "Available disk space: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 25 ]; then
  echo "ERROR: Need at least 25 GB free (FP4 is ~22.81 GB)"
  exit 1
fi
echo

# Download using Python API
python3 << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os
import sys

hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
cache_dir = os.path.join(hf_home, 'hub')
os.makedirs(cache_dir, exist_ok=True)

try:
    print("Downloading JoyAI-Echo FP4...")
    snapshot_download(
        repo_id="jdopensource/JoyAI-Echo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["echo15_fp4*"],
    )
    print("✓ Download complete!")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
  ECHO_CACHE="$HF_HOME/hub/models--jdopensource--JoyAI-Echo"
  SIZE=$(du -sh "$ECHO_CACHE" 2>/dev/null | cut -f1)
  echo
  echo "JoyAI-Echo FP4 installed: $SIZE"
  echo "Location: $ECHO_CACHE"
  echo
  echo "Ready to run:"
  echo "  export JOYOMNI_MODEL=echo_fp4"
  echo "  bash run_server_fp4.sh"
else
  echo "Download failed"
  exit 1
fi
