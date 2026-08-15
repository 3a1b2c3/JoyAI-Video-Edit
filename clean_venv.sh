#!/bin/bash
# Clean up and recreate virtual environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Cleaning Virtual Environment"
echo "========================================================================"
echo ""

if [ -d ".venv" ]; then
    echo "Removing corrupted .venv..."
    rm -rf .venv
    echo "  ✅ Removed"
else
    echo "  ℹ .venv not found (nothing to clean)"
fi

echo ""
echo "Creating fresh virtual environment..."
bash setup_venv.sh

echo ""
echo "========================================================================"
echo "✅ Clean venv ready!"
echo "========================================================================"
