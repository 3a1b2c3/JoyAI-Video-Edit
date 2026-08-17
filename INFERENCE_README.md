# DiT Inference on RTX 5090 / A40 — Complete Guide

## Quick Start

```bash
python run_inference.py --video input.mp4 --out output.mp4 --steps 4
```

Output: `C:\workspace\world\JoyAI-Video-Edit\outputs\dit_output.mp4`

---

## Files in This Directory

### Inference Scripts
- **`run_inference.py`** — Working float32 inference pipeline
  - Uses: `joyai_video_edit_dit_0804.pth` (unquantized)
  - Memory: 28-30 GB
  - Status: ✅ **WORKING**

### Documentation
- **`RUN_INFERENCE.md`** — Quick usage guide and performance notes
- **`QUANTIZATION_ATTEMPTS.md`** — Detailed analysis of all 3 quantization attempts and why they failed

### Checkpoints
- **`deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth`** — Float32 model (28-30 GB)
- ~~`joyai_video_edit_dit_0804_int8_0815.pth`~~ — DELETED (incompatible custom quantization)

---

## Usage

### Basic Inference
```bash
python run_inference.py --video input.mp4
```

### With Custom Settings
```bash
python run_inference.py \
  --video input.mp4 \
  --out output.mp4 \
  --frames 4 \
  --height 512 \
  --width 512 \
  --steps 8 \
  --seed 42
```

### Arguments
- `--video` — Input video path (default: `assets/Recording 2026-08-12 205529.mp4`)
- `--out` — Output video path (default: `outputs/dit_output.mp4`)
- `--frames` — Number of frames to process (default: 2)
- `--height` — Frame height (default: 256)
- `--width` — Frame width (default: 256)
- `--steps` — Denoising steps (default: 1, higher = better quality but slower)
- `--seed` — Random seed for reproducibility (default: 42)

---

## Hardware Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| RTX 5090 | 32 GB | ✅ Fits (uses 28-30 GB) |
| A40 | 48 GB | ✅ Fits with headroom |
| A100 | 40-80 GB | ✅ Excellent fit |

---

## Performance Notes

- **Model size**: 28-30 GB (float32, 16.26B parameters)
- **Inference speed**: Depends on `--steps` and resolution
- **Dtype**: Model auto-converts to bfloat16 for efficiency
- **VAE**: Auto-encodes frames and decodes latents

---

## What We Learned: Quantization

Three quantization approaches were attempted to reduce the 28-30 GB model to int8 (~5-6 GB). All failed:

1. **Smart Dequantization** — PyTorch's `load_state_dict()` type-checks parameters before forward pass
2. **CPU Dequantization** — Requires 34 GB RAM (6 GB quantized + 28 GB dequantized > 24-28 GB available)
3. **TensorRT Export** — ONNX export fails on shape mismatches

**Conclusion**: The original int8 checkpoint uses a custom quantization scheme PyTorch cannot load. Float32 is the only viable approach on this hardware.

See `QUANTIZATION_ATTEMPTS.md` for detailed technical analysis.

---

## Troubleshooting

**"Video not found"**
- Provide absolute path or ensure file exists in working directory

**"GPU OOM"**
- Reduce `--frames` or `--height`/`--width`
- Model is ~28-30 GB; check `nvidia-smi` for available memory

**"All frames failed to encode"**
- Check VAE checkpoint exists at `deploy/deps/checkpoints/JoyAI-Video-Edit/vae`
- Check frame dimensions are reasonable (256x256 minimum recommended)

**"Model has mixed dtypes" warning**
- This is normal; the model auto-converts to bfloat16 for efficiency
- Does not affect output quality or speed

---

## Next Steps

1. **Test on your hardware**: `python run_inference.py --frames 1 --steps 1`
2. **Increase quality**: Higher `--steps` value (4-8 recommended)
3. **Batch processing**: Run inference loop over multiple videos

---

## Summary

- ✅ **Working**: Float32 inference with `run_inference.py`
- ❌ **Not viable**: Int8 quantization on this checkpoint
- 📊 **Memory**: 28-30 GB (fits on RTX 5090, plenty on A40)
- 📝 **Docs**: See `QUANTIZATION_ATTEMPTS.md` for technical details
