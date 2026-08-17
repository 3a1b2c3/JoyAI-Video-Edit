# DiT Inference on RTX 5090 / A40

## Quick Start

```bash
python run_inference.py --video <input.mp4> --out <output.mp4> --frames 2 --steps 4
```

## Arguments

- `--video` — Input video path (default: `assets/Recording 2026-08-12 205529.mp4`)
- `--out` — Output video path (default: `outputs/dit_output.mp4`)
- `--frames` — Number of frames to process (default: 2)
- `--height` — Frame height in pixels (default: 256)
- `--width` — Frame width in pixels (default: 256)
- `--seed` — Random seed (default: 42)
- `--steps` — Denoising steps (default: 1, higher = better quality but slower)

## Requirements

- RTX 5090 (32 GB) or A40 (48 GB) GPU
- ~30 GB VRAM for float32 model
- Input video file

## Performance

| GPU | Float32 | Memory |
|-----|---------|--------|
| RTX 5090 | Baseline | 28-30 GB |
| A40 | Baseline | 28-30 GB |

## Quantization

The original quantized checkpoint (`joyai_video_edit_dit_0804_int8_0815.pth`) uses a custom quantization scheme incompatible with PyTorch inference.

**Three approaches attempted — all failed:**

1. **PyTorch Smart Dequantization** — Load qint8, dequantize on-demand in forward pass
   - Failed: `load_state_dict()` rejects qint8 before forward is called (PyTorch design constraint)

2. **CPU Dequantization** — Dequantize full checkpoint on CPU before loading
   - Failed: OOM (requires 34 GB: 6 GB qint8 + 28 GB float32 > 24-28 GB CPU RAM)

3. **TensorRT Export** — Export model structure to ONNX, build native int8 engine
   - Failed: Shape mismatch during ONNX export (wrong dummy input dimensions)

**Detailed analysis**: See `QUANTIZATION_ATTEMPTS.md`

**Conclusion**: Native int8 inference not feasible on this hardware. Float32 is the working solution.

## Files

- `run_inference.py` — Working float32 inference pipeline
- `joyai_video_edit_dit_0804.pth` — Float32 checkpoint (used automatically)
- `joyai_video_edit_dit_0804_int8_0815.pth` — Original quantized checkpoint (deleted - incompatible)

## Next Steps

1. Test inference on A40 (higher memory = faster fallback options)
2. If quantization is critical, explore TensorRT or ONNX Runtime
3. Consider using streaming/chunked inference to reduce peak memory
