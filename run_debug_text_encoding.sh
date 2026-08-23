#!/bin/bash
# Debug: Compare random vs text-encoded context

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONPATH="$PWD/deploy/joyomni_ops:$PWD/deploy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64:$(python3 -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"

mkdir -p "$TORCH_HOME" "$HF_HOME"

echo "Debugging text encoding..."
echo ""

python3 -u debug_text_encoding.py "$@"
