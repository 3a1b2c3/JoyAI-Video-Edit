#!/bin/bash
# Build joyomni_ops against this repo's own .venv, explicitly -- not via `which
# python`, which picks up a DIFFERENT project's venv (scope/.venv) leaking in through
# WSL's Windows-PATH interop ahead of this repo's activated venv.
#
#   wsl bash build_joyomni_local.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HERE/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found or not executable." >&2
    exit 1
fi

echo "Using interpreter: $PYTHON ($("$PYTHON" --version 2>&1))"
cd "$HERE/deploy/joyomni_ops"
PYTHON="$PYTHON" bash build.sh
