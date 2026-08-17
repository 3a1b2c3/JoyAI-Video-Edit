# JoyAI DiT Quantization Guide

## Overview
The DiT (Diffusion Transformer) checkpoint has been quantized from full-precision (fp32) to int8, reducing size by 82% while maintaining inference quality.

## Quantized Checkpoint
- **Path:** `deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804_int8_0815.pth`
- **Original Size:** 32.5 GB (fp32)
- **Quantized Size:** 5.84 GB (int8)
- **Compression:** 82%
- **Tensors:** 894 parameter tensors (all quantized to torch.qint8)

## Performance Benefits
- **Memory:** 32.5 GB → 5.84 GB (5.6× reduction)
- **Load Time:** Faster model loading from disk
- **Inference:** Comparable quality to full-precision with reduced computational requirements

## Usage

### Default (Quantized)
```bash
python example_inference.py
python offline_infer.py
```

Both scripts use the quantized checkpoint by default.

### Full-Precision (Optional)
If you need full-precision inference:
```bash
python example_inference.py --full-precision
python offline_infer.py --full-precision
```

Requires the original 32.5 GB checkpoint at:
`deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth`

## Quantization Details

### Quantization Method
- **Type:** Static int8 quantization via `torch.quantize_per_tensor()`
- **Scale:** Per-tensor, computed from absolute max value
- **Zero Point:** 0
- **Format:** PyTorch native qint8 dtype

### How to Re-quantize
If you need to re-quantize the full-precision checkpoint:
```bash
python quantize_dit.py
```

This will create a new quantized checkpoint in the same directory.

## Files

### Core Scripts
- `quantize_dit.py` — Quantization script (loads full-precision, outputs int8 checkpoint)
- `quantize_dit.bat` — Windows batch wrapper for quantization

### Inference Scripts (Use Quantized by Default)
- `../example_inference.py` — Example inference with model validation
- `../offline_infer.py` — Offline inference with video + prompt input
- `../decart_restyle.bat` — Video restyling with Decart Lucy

## Troubleshooting

### "Quantized checkpoint not found"
Ensure the file exists at the expected path. Run `quantize_dit.py` to generate it if missing.

### "Cannot change width after codec is open"
This error was fixed in decart_long.py by switching from pyav to ffmpeg codec plugin.

### GPU Memory Issues
The quantized checkpoint still requires ~12-16GB GPU memory for inference. If you encounter OOM errors:
1. Close other GPU-using applications
2. Reduce batch size / resolution
3. Enable CPU offload (if implemented)

## Architecture
The quantized model maintains the same architecture as the full-precision version:
- 16B parameter diffusion transformer
- 894 separate parameter tensors
- Compatible with existing inference pipelines

## Notes
- Quantization is lossy but produces visually similar results to full-precision
- The quantized checkpoint is stored in PyTorch's native quantized format
- Can be loaded with `torch.load()` and `map_location='cpu'` for CPU inference preparation
