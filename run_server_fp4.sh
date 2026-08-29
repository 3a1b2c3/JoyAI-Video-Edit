#!/bin/bash
# Start JoyAI-Video-Edit server with Echo FP4 + FP8 disabled + low-VRAM for stability
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/server.log"

{
  echo "=== Server startup: $(date) ==="
  echo "Script dir: $SCRIPT_DIR"
  echo "Log: $LOG_FILE"
  echo

  echo "Setting environment..."
  export JOYOMNI_MODEL=echo_fp4
  export JOYOMNI_FP8_IMG=0
  export JOYOMNI_FP8_TXT=0
  export JOYOMNI_LOW_VRAM=1
  echo "JOYOMNI_MODEL=$JOYOMNI_MODEL"
  echo "JOYOMNI_FP8_IMG=$JOYOMNI_FP8_IMG"
  echo "JOYOMNI_FP8_TXT=$JOYOMNI_FP8_TXT"
  echo "JOYOMNI_LOW_VRAM=$JOYOMNI_LOW_VRAM"
  echo

  echo "Starting run_server_best.sh..."
  bash run_server_best.sh "$@"
} | tee -a "$LOG_FILE" 2>&1
