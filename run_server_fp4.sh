#!/bin/bash
# Start JoyAI-Video-Edit server with Echo FP4 + FP8 disabled
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting JoyAI-Video-Edit server with Echo FP4 (no FP8)..."
echo

export JOYOMNI_FP8_IMG=0
export JOYOMNI_FP8_TXT=0
bash run_server_best.sh "$@"
