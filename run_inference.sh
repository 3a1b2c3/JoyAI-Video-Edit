#!/bin/bash
# Run DiT inference - no venv required

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use system Python (no venv)
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# Debug: Show initial memory
echo "[DEBUG] Initial memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits | sed 's/,/ \/ /g' | sed 's/^/  GPU: /' | sed 's/$/ MB/'
free -h | grep Mem | awk '{print "  RAM: " $3 " used / " $2 " total"}'
echo ""

# Parse arguments (handle spaces in filenames)
VIDEO="${1:-assets/Recording 2026-08-12 205529.mp4}"
OUTPUT="${2:-outputs/dit_output.mp4}"
FRAMES="${3:-2}"
HEIGHT="${4:-256}"
WIDTH="${5:-256}"
STEPS="${6:-1}"

# Run inference with properly quoted args + memory monitoring
echo "[DEBUG] Running inference..."
(while true; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits | sed 's/,/ \/ /g' | sed 's/^/  [GPU] /' | sed 's/$/ MB/'; free -h | grep Mem | awk '{print "  [RAM] " $3 " \/ " $2}'; sleep 1; done) &
MONITOR_PID=$!

$PYTHON run_inference.py \
  --video "$VIDEO" \
  --out "$OUTPUT" \
  --frames "$FRAMES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS"

RESULT=$?
kill $MONITOR_PID 2>/dev/null

echo ""
echo "[DEBUG] Final memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits | sed 's/,/ \/ /g' | sed 's/^/  GPU: /' | sed 's/$/ MB/'
free -h | grep Mem | awk '{print "  RAM: " $3 " used / " $2 " total"}'

exit $RESULT

echo ""
echo "✅ Inference complete!"
echo "Output: $(cd outputs && pwd)/dit_output.mp4"
