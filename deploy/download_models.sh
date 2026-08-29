#!/bin/bash
# Wrapper that calls Python download script instead of using hf CLI
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 download_models.py
exit $?
