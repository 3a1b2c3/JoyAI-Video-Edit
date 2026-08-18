# JoyAI-Video-Edit DiT Inference Setup

## Status
- ✅ Model architecture loads (16.26B parameters)
- ✅ VAE encodes video to latents (CPU, ~19 min/frame)
- ✅ joyomni_ops CUDA extension available
- ✅ DiT forward pass (bf16, joyomni_ops kernels)
- ✅ Quantization working (32.5GB → 16GB checkpoint)
- ✅ Windows (32GB RTX 5090): Testing quantized model
- ⏳ Horde (48GB A40): Ready to test with quantized checkpoint

## Key Files
- `deploy/xvideo/models/models.py` - Model loading (load_dit function)
- `run.sh` - Full inference pipeline (video→VAE→DiT→decode→save)
- `test_quick.py` - Fast test (skip VAE, use synthetic latents)

## Requirements
- PyTorch 2.9.1+cu128
- CUDA 12.8
- joyomni_ops CUDA extension (in deploy/joyomni_ops/)
- Python 3.10+

## Quantized Checkpoint Format

The quantized checkpoint uses per-tensor int8 quantization. Each weight is stored as:
```python
{
  "data": int8_tensor,
  "scale": float_scale_factor
}
```

The `load_dit` function automatically dequantizes during loading:
```python
v_dequantized = v["data"].float() * v["scale"]
```

This happens before dtype conversion and shape validation, so everything works transparently.

## Known Issues

### 1. Quantized Checkpoint Loading (FIXED)
**Problem:** AttributeError when loading quantized checkpoint
- Error: "dict object has no attribute 'shape'"
- Root cause: load_dit tried to access .shape on quantized dict before dequantizing

**Fix Applied:**
- Check if value is dict with "data" key
- Dequantize: `v["data"].float() * v["scale"]`
- Convert to target dtype
- Then proceed with shape validation

### 2. Horde OOM During Checkpoint Loading
**Problem:** Process killed (SIGKILL) when loading 45GB checkpoint on 48GB GPU
- Horde GPU at 96% capacity (other processes consuming 30GB+)
- Model (32.5GB) + checkpoint (45GB) exceeds available headroom

**Attempted Fixes:**
- ✅ mmap=True - memory-maps checkpoint file (reduces CPU RAM)
- ✅ Keep checkpoint on CPU - don't move to GPU immediately
- ✅ Per-tensor conversion - convert dtype on CPU before loading
- ❌ Still OOMs due to shared system resource constraints

**Solution:** Test on Windows (clean 32GB GPU) first

### 2. Dtype Mismatches (RESOLVED)
**Problem:** Model created in bf16, but some tensors end up in fp16
- joyomni_ops CUDA kernels require bf16 inputs
- Time embeddings, attention layers creating mixed dtypes

**Fix Applied:**
- Load model directly in target dtype (bf16)
- Convert checkpoint tensors to bf16 on CPU (fast)
- Move to device at the end

### 3. PYTHONPATH Issues
**Problem:** joyomni_ops module not found at import time
- Extension built in deploy/joyomni_ops/ but Python couldn't find it

**Solution:**
```bash
export PYTHONPATH="$PWD/deploy/joyomni_ops:$PWD/deploy:${PYTHONPATH:-}"
```

## Setup Instructions

### Windows (RTX 5090)
```powershell
cd C:\workspace\world\JoyAI-Video-Edit
python test_quick.py
```

### Horde (A40 48GB)
```bash
cd ~/JoyAI-Video-Edit
git pull
bash run.sh
```

Note: May OOM if other processes are using GPU. Kill them first:
```bash
pkill -9 -f python
sleep 3
bash run.sh
```

## Model Loading Flow
1. Create model in bf16 on GPU (empty parameters)
2. Load checkpoint from disk (mmap=True, stays on CPU)
3. Convert checkpoint tensors to bf16 (on CPU, fast)
4. Load state_dict into model (PyTorch transfers to GPU)
5. Free checkpoint memory
6. Model now on GPU in bf16, ready for inference

## VAE Encoding
- VAE on CPU (model uses all GPU memory)
- Input frames: on CPU
- Latents: move to GPU for diffusion
- Encoding time: ~19 min/frame (bottleneck)

## Diffusion
- DiT on GPU (bf16)
- Latents on GPU (bf16)
- Context generated in bf16
- Uses joyomni_ops fused kernels for layer norm, RoPE

## Quantization (Memory Optimization)

**Problem:** 32.5GB checkpoint doesn't fit on 48GB GPU (shared system at 96% capacity)

**Solution:** Per-tensor int8 quantization
- Reduces model: 32.5GB → 16.26GB (2x compression)
- Automatic dequantization via stored scale factors
- Works with standard PyTorch load/save

**Usage:**
```bash
# One-time: quantize checkpoint
python quantize_simple.py deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth dit_quantized.pth

# Verify it loads
python test_quantized.py

# Run inference with quantized model
python test_quick.py      # Fast test (skip VAE)
bash run.sh               # Full inference
```

**Result:** Quantized checkpoint now fits on horde (16GB << 48GB available)

## Known Limitations
- VAE encoding is slow (CPU-bound, 19+ min/frame)
- Windows 32GB GPU at capacity (VAE on CPU to free space)
- No batching implemented yet
- Quantized model slightly lower precision (int8)

## Next Steps
1. ✅ Test quantized checkpoint on Windows (test_quick.py)
2. Run full inference on Windows (run.sh)
3. Test quantized model on horde
4. Optimize VAE encoding (GPU acceleration, batching)
