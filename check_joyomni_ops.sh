#!/bin/bash
# Prerequisite check: verify joyomni_ops CUDA kernels import cleanly for the
# interpreter that will run inference. Run this before infer_standalone.sh /
# run_server*.sh -- catches a stale or wrong-Python-version build early
# instead of failing mid-generation.
# Usage: bash check_joyomni_ops.sh [python_bin]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${1:-python}"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: interpreter '$PYTHON' not found." >&2
    echo "  Usage: bash check_joyomni_ops.sh [python_bin]" >&2
    exit 1
fi

"$PYTHON" deploy/check_joyomni_ops.py
