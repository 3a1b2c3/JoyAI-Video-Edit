# JoyAI Video Edit - Quantized Model Setup (Start Over)

This guide gets you running JoyAI Video Edit with **PyTorch native quantization** (int8 dynamic) instead of full-precision fp32.

## Summary

**Assets you have:**
- Input video: `assets/Recording 2026-08-12 205529.mp4`
- Edit prompt: `assets/decart_prompt.txt` (neon racing track)
- FP32 checkpoint: `deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth` (downloaded)

**Quantization approach:**
- Quantize yourself using PyTorch native `torch.quantization.quantize_dynamic()` ← **Most reliable**
- Alternative: Download pre-quantized (if available, but may have load issues)

## Step 1a: Quantize the Checkpoint (Recommended)

You have the fp32 checkpoint (0811.pth). Quantize it to int8 using PyTorch native quantization:

```bash
cd C:\workspace\world\JoyAI-Video-Edit
python quantize_pytorch_native.py --method dynamic
```

This will:
1. Load fp32 checkpoint (32.5 GB)
2. Apply dynamic int8 quantization (no calibration needed)
3. Save quantized version (5.84 GB, 82% smaller)
4. **Validate** by reading back to confirm format is correct

**Output:** `deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811_quantized.pth`

## Step 1b: Or Download Pre-Quantized (Optional, May Have Load Issues)

If you prefer not to quantize:
```
Download: https://huggingface.co/jdopensource/JoyAI-Video-Edit
Path: deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0815_int8.pth
```

⚠️ Pre-quantized may use different format. If load fails, fall back to Step 1a.

## Step 2: Set Up Virtual Environment

```bash
cd C:\workspace\world\JoyAI-Video-Edit
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit
python -m pip install -r deploy/requirements.txt
```

## Step 3: Run Inference

### Option A: Windows Batch (Easiest)
```bash
run_quantized.bat
```

This uses:
- Quantized checkpoint (default)
- Input: `assets/Recording 2026-08-12 205529.mp4`
- Prompt: `assets/decart_prompt.txt`
- Output: `outputs/joyai_quantized_output.mp4`

### Option B: Direct Python (Full Control)
```bash
python run_quantized_inference.py \
    --video assets/Recording\ 2026-08-12\ 205529.mp4 \
    --prompt assets/decart_prompt.txt \
    --out outputs/my_output.mp4 \
    --frames 10 \
    --height 512 \
    --width 512 \
    --steps 4
```

**Available arguments:**
```
--video         Input video path (default: assets/Recording 2026-08-12 205529.mp4)
--prompt        Edit prompt file or text (default: assets/decart_prompt.txt)
--out           Output path (default: outputs/joyai_quantized_output.mp4)
--frames        Frames to process (default: 2)
--height        Frame height (default: 256)
--width         Frame width (default: 256)
--seed          Random seed (default: 42)
--steps         Diffusion steps (default: 1)
--fp32          Use fp32 instead of quantized (not recommended)
```

### Option C: Full-Precision Fallback
If you need fp32 (not recommended due to memory):
```bash
python run_quantized_inference.py --fp32
```

## Key Improvements

| Metric | FP32 | INT8 (Quantized) |
|--------|------|-----------------|
| Size | 32.5 GB | 5.84 GB |
| Memory | ~50 GB | ~12-16 GB |
| Load time | Slow | Fast |
| Inference quality | Full precision | Comparable (lossy) |
| **Recommended** | ❌ | ✓ |

## Troubleshooting

### "Checkpoint not found"
1. Download from HuggingFace: https://huggingface.co/jdopensource/JoyAI-Video-Edit
2. Place at: `deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0815_int8.pth`
3. Re-run

### "Video not found"
Ensure video path is correct:
```
assets/Recording 2026-08-12 205529.mp4
```

### "Out of memory"
- Reduce `--frames` (default: 2)
- Reduce `--height` or `--width` (default: 256×256)
- Close other GPU applications
- Use `--fp32` with CPU offload (if implemented)

### "CUDA not available"
Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Output

Video is saved at: `outputs/joyai_quantized_output.mp4`

With your assets:
- Prompt: "A high-speed racing track. Vibrant neon colors..."
- Output will apply the neon racing aesthetic to your input video

## What's Quantized?

**PyTorch Native Dynamic Int8 Quantization:**
- **Standard API**: `torch.quantization.quantize_dynamic()` (built-in, reliable)
- **894 parameter tensors** quantized: fp32 → qint8
- **Dequantization**: Automatic during forward pass (transparent, no manual step)
- **Save/Load**: Standard `torch.save()` / `torch.load()` (no custom format issues)
- **Memory**: 5.84 GB quantized + auto-dequant at runtime (no 34 GB RAM requirement)

**Why this works** (unlike pre-quantized checkpoints):
- Pre-quantized uses custom quantization scheme → breaks with `load_state_dict()`
- PyTorch native uses standard format → `load_state_dict()` works, dequantization automatic
- Avoids the 3 known failures: smart-dequant (type mismatch), CPU-dequant (OOM), TensorRT (shape errors)

## Files

- `run_quantized_inference.py` — Full inference pipeline (quantized by default)
- `run_quantized.bat` — Windows batch launcher
- `assets/Recording 2026-08-12 205529.mp4` — Your input video
- `assets/decart_prompt.txt` — Your edit prompt
- `outputs/joyai_quantized_output.mp4` — Generated output

## Troubleshooting: Quantization Issues

### "Cannot load quantized checkpoint"
If you downloaded pre-quantized and get:
```
RuntimeError: Copying from quantized Tensor to non-quantized Tensor is not allowed
```

**Root cause**: Pre-quantized uses custom quantization scheme that conflicts with PyTorch's `load_state_dict()`.

**Solution**: Use PyTorch native quantization instead (step 1a above):
```bash
python quantize_pytorch_native.py --method dynamic
```

### "Out of memory during dequantization"
If you try to dequantize the whole model at once:
```
MemoryError: Cannot allocate 28 GB
```

**Root cause**: Holding both qint8 (6 GB) + float32 (28 GB) = 34 GB needed, only have 24-28 GB CPU RAM.

**Solution**: Use PyTorch native quantization — it dequantizes on-the-fly during inference (no need to hold both).

## Next Steps

1. **Quantize:** `python quantize_pytorch_native.py --method dynamic`
2. **Run:** `run_quantized.bat` or `python run_quantized_inference.py`
3. **Check:** `outputs/joyai_quantized_output.mp4`
