#!/bin/bash
# Debug runner - trace what the model is doing

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/deploy/joyomni_ops:${PYTHONPATH:-}"

python3 debug_inference.py "$@"
