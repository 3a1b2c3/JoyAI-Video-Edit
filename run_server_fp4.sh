#!/bin/bash
# Start JoyAI-Video-Edit server with Echo FP4 + FP8 disabled. Low-VRAM mode
# is decided by run_server_best.sh's GPU auto-detection, not forced here.
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
  # JOYOMNI_LOW_VRAM is intentionally left unset here -- run_server_best.sh
  # (invoked below) auto-detects the GPU and picks the right value per
  # DEPLOYMENT.md (e.g. 0 on >48 GiB cards, 1 on <=48 GiB). Hardcoding =1
  # here used to shadow that detection unconditionally, forcing low-VRAM
  # mode's block-by-block DiT staging and text-encoder CPU offload even on
  # large cards with no VRAM pressure. Still overridable by exporting
  # JOYOMNI_LOW_VRAM before calling this script.
  echo "JOYOMNI_MODEL=$JOYOMNI_MODEL"
  echo "JOYOMNI_FP8_IMG=$JOYOMNI_FP8_IMG"
  echo "JOYOMNI_FP8_TXT=$JOYOMNI_FP8_TXT"
  echo

  echo "Starting run_server_best.sh..."
  bash run_server_best.sh "$@"
} | tee -a "$LOG_FILE" 2>&1
