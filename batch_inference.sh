#!/bin/bash
# Batch process multiple videos through DiT inference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parameters
INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-outputs/batch}"
FRAMES="${3:-2}"
STEPS="${4:-1}"
PATTERN="${5:-*.mp4}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Batch DiT Inference"
echo "========================================================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Frames per video: $FRAMES"
echo "Denoising steps:  $STEPS"
echo "File pattern:     $PATTERN"
echo "========================================================================"
echo ""

# Find and process videos
VIDEOS=$(find "$INPUT_DIR" -maxdepth 1 -name "$PATTERN" -type f | sort)
COUNT=$(echo "$VIDEOS" | wc -l)

if [ "$COUNT" -eq 0 ]; then
    echo "❌ No videos found matching: $INPUT_DIR/$PATTERN"
    exit 1
fi

echo "Found $COUNT video(s)"
echo ""

INDEX=1
for VIDEO in $VIDEOS; do
    BASENAME=$(basename "$VIDEO")
    OUTFILE="$OUTPUT_DIR/${BASENAME%.*}_dit.mp4"

    echo "[$INDEX/$COUNT] Processing: $BASENAME"
    echo "  Output: $(basename "$OUTFILE")"

    python run_inference.py \
      --video "$VIDEO" \
      --out "$OUTFILE" \
      --frames "$FRAMES" \
      --steps "$STEPS" 2>&1 | grep -E "✓|✅|ERROR" || true

    if [ -f "$OUTFILE" ]; then
        SIZE=$(du -h "$OUTFILE" | cut -f1)
        echo "  ✅ Complete ($SIZE)"
    else
        echo "  ❌ Failed"
    fi

    echo ""
    INDEX=$((INDEX + 1))
done

echo "========================================================================"
echo "✅ Batch processing complete!"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"
