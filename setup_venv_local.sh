#!/bin/bash
# Create/refresh this repo's .venv (WSL-native, matches what run_server_local.bat /
# build_joyomni_local.sh expect) and install the base deps from deploy/requirements.txt.
# Plain stdlib venv/pip.
#
#   wsl bash setup_venv_local.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY_BIN="python3.12"
if ! command -v "$PY_BIN" &>/dev/null; then
    echo "ERROR: $PY_BIN not found. Install Python 3.12 first (matches the existing local setup)." >&2
    exit 1
fi

echo "=== Creating .venv (python 3.12, matching the existing local setup) ==="
"$PY_BIN" -m venv .venv

echo
echo "=== Installing base dependencies (deploy/requirements.txt) ==="
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r deploy/requirements.txt

echo
echo "Done. Next: wsl bash build_joyomni_local.sh   (builds the joyomni_ops CUDA extension)"
