#!/bin/bash
# Full inference runner - load video, encode, diffuse, decode, save

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD/deploy/joyomni_ops:${PYTHONPATH:-}"

python3 run_module.py "$@"
