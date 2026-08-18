#!/bin/bash
# Quick test runner - synthetic latents, skip VAE encoding

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD/deploy/joyomni_ops:${PYTHONPATH:-}"

python3 test_quick_module.py "$@"
