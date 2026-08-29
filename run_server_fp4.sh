#!/bin/bash
# Start JoyAI-Video-Edit server with Echo FP4 + FP8 disabled + low-VRAM for stability
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting JoyAI-Video-Edit server with Echo FP4 (no FP8, low-VRAM)..."
echo

export JOYOMNI_MODEL=echo_fp4
export JOYOMNI_FP8_IMG=0
export JOYOMNI_FP8_TXT=0
export JOYOMNI_LOW_VRAM=1
bash run_server_best.sh "$@"
