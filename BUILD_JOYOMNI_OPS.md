# Building joyomni_ops Locally

## Prerequisites
- CUDA Toolkit 12.4+ (with nvcc compiler)
- PyTorch with CUDA support
- Python 3.10+

## Build Steps

### 1. Install CUDA Toolkit (if not installed)
```bash
# On Windows: Download from https://developer.nvidia.com/cuda-toolkit
# On Linux: sudo apt install nvidia-cuda-toolkit
```

### 2. Verify CUDA Setup
```bash
nvcc --version
echo $CUDA_HOME
```

### 3. Build joyomni_ops
```bash
cd deploy/joyomni_ops

# Set environment variables
export CUDA_HOME=/usr/local/cuda  # or your CUDA installation path
export JOYOMNI_OPS_NO_FP8=1       # Skip FP8 GEMM compilation (optional, speeds up build)

# Build and install
python setup.py build_ext --inplace
pip install -e .
```

### 4. Verify Installation
```bash
python -c "from joyomni_ops import fused_norm_scale_shift; print('✅ joyomni_ops installed')"
```

## Environment Variables

- **`CUDA_HOME`**: Path to CUDA installation (e.g., `/usr/local/cuda-12.4`)
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.4
  ```

- **`JOYOMNI_OPS_NO_FP8=1`**: Skip FP8 GEMM compilation during build
  - Faster build time (skips CUTLASS FP8 kernel compilation)
  - FP8 quantization features disabled in runtime
  - **Recommended**: Set this unless you specifically need FP8 support
  ```bash
  export JOYOMNI_OPS_NO_FP8=1
  ```

- **`JOYOMNI_OPS_CUTLASS_DIR`**: Path to CUTLASS library (optional, for FP8 support)
  - Only needed if building with FP8 support (without `JOYOMNI_OPS_NO_FP8=1`)
  ```bash
  export JOYOMNI_OPS_CUTLASS_DIR=/path/to/cutlass
  ```

## Troubleshooting

### nvcc not found
```bash
export PATH=$CUDA_HOME/bin:$PATH
```

### CUDA_HOME not set
```bash
export CUDA_HOME=/usr/local/cuda-12.4
```

### Build fails on missing headers
Ensure CUDA development tools are installed (not just runtime).

## After Build

Once built, inference will automatically use joyomni_ops for fused operations:
```bash
python run_inference.py --frames 1 --height 256 --width 256 --steps 1
```
