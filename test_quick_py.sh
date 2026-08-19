#!/bin/bash
# Quick test runner - synthetic latents, skip VAE encoding (uses joyomni_ops)

cd "$(dirname "${BASH_SOURCE[0]}")"

# Setup environment
export PYTHONPATH="$PWD/deploy/joyomni_ops:$PWD/deploy:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64:$(python3 -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"

python3 -u test_quick_module.py "$@"
