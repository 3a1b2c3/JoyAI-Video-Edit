# Quantized DiT Inference: Wrapper Model Approach

## Overview

Use a quantized-aware model wrapper that loads qint8 tensors directly without conversion to float32.

**Key advantage**: No type mismatch, no 2x memory spike, tensors stay quantized throughout.

---

## Architecture

### QuantizedTransformer3DModel (New)

**File**: `deploy/xvideo/models/quantized_dit.py`

**Design**: Inherits from Transformer3DModel, overrides weight loading

**How it works**:
1. Detect quantized tensors in checkpoint
2. Store qint8 tensors as buffers (not parameters) to bypass dtype validation
3. Forward pass unchanged (PyTorch ops handle qint8 natively)
4. No dequantization, no conversion

**Key methods**:
- `load_state_dict()` - Accept qint8 tensors as buffers
- `forward()` - Standard inference with qint8 weights

### Modified load_dit() in models.py

**Changes**:
1. Detect if checkpoint has quantized tensors
2. Use QuantizedTransformer3DModel if qint8 detected
3. Otherwise use standard Transformer3DModel
4. Keep CPU-first instantiation (prevents OOM)

**Flow**:
```
Load checkpoint
    ↓
Check: has quantized tensors?
    ↓
YES → Use QuantizedTransformer3DModel
     → Load state_dict (buffers for qint8)
     → Move to GPU
    ↓
NO  → Use standard Transformer3DModel
     → Load state_dict normally
     → Move to GPU
```

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| QuantizedTransformer3DModel | ✓ Created | `deploy/xvideo/models/quantized_dit.py` |
| load_dit() detection | ✓ Updated | `deploy/xvideo/models/models.py` |
| models.py imports | ✓ Added | `deploy/xvideo/models/models.py` |
| Inference scripts | ⏳ Ready | `joyai_quantized.py` (uses updated load_dit) |

---

## Execution Plan

### Phase 1: Test Checkpoint Loading

**Goal**: Verify quantized checkpoint loads without errors

```bash
cd c:\workspace\world\JoyAI-Video-Edit

python -c "
import sys; sys.path.insert(0, 'deploy')
import torch
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

cfg = ExpConfig()
cfg.dit_ckpt = 'deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804_int8_0815.pth'
device = torch.device('cuda')

print('Loading quantized DiT...')
model = load_dit(cfg, device)
print(f'✓ Loaded successfully')
print(f'  Model class: {model.__class__.__name__}')
print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
"
```

**Expected output**:
```
Loading DiT checkpoint: ...joyai_video_edit_dit_0804_int8_0815.pth
Detected quantized tensors in checkpoint - using QuantizedTransformer3DModel
✓ Loaded successfully
  Model class: QuantizedTransformer3DModel
  Parameters: 16.3B
```

### Phase 2: Test Single Inference

**Goal**: Run one forward pass with quantized model

```bash
python -c "
import sys; sys.path.insert(0, 'deploy')
import torch
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

cfg = ExpConfig()
cfg.dit_ckpt = 'deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804_int8_0815.pth'
device = torch.device('cuda')

model = load_dit(cfg, device)
model.eval()

# Create dummy inputs
batch_size = 1
x = torch.randn(batch_size, 64, 1, 16, 16, device=device, dtype=torch.float32)
t = torch.randint(0, 1000, (batch_size,), device=device)
context = torch.randn(batch_size, 256, 4096, device=device, dtype=torch.bfloat16)

print('Running single forward pass...')
with torch.no_grad():
    output = model(x, t, context=context)
print(f'✓ Forward pass complete')
print(f'  Output shape: {output.shape}')
print(f'  Output dtype: {output.dtype}')
"
```

**Expected output**:
```
Running single forward pass...
✓ Forward pass complete
  Output shape: torch.Size([1, 64, 1, 16, 16])
  Output dtype: torch.float32
```

### Phase 3: Full Inference Pipeline

**Goal**: Generate complete video

```bash
python joyai_quantized.py \
  --frames 4 \
  --steps 2 \
  --height 256 \
  --width 256 \
  --out outputs/test_quantized.mp4
```

**Monitor**:
- Model load time
- Inference speed
- GPU memory peak
- Output video validity

### Phase 4: Scale Resolution (if Phase 3 succeeds)

```bash
python joyai_quantized.py \
  --frames 8 \
  --steps 4 \
  --height 512 \
  --width 512 \
  --out outputs/quantized_512.mp4
```

---

## Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| Checkpoint file | 5.84 GB | On-disk size (qint8) |
| GPU memory peak | ~12-16 GB | Model + inference |
| Type mismatch error | ✓ Resolved | No conversion needed |
| Inference speed | 1-2 fps | Estimated |
| Video quality | High | qint8 preserves precision well |

---

## Memory Analysis

**GPU: 32 GB total**

```
Quantized model (qint8):    ~6 GB
Float buffers (activations): ~8 GB
VAE (fp16):                 ~2 GB
Inference overhead:         ~3 GB
                           ------
Total:                      ~19 GB (comfortable margin)
```

**vs Full-precision path**:
```
Float32 model:              ~28 GB
+ activations:              ~2 GB
+ VAE:                      ~2 GB
                           ------
Total:                      ~32 GB (tight, may OOM)
```

**Quantized is significantly more efficient.**

---

## Troubleshooting

### Issue: "QuantizedTransformer3DModel not found"
**Cause**: Python path issue or import failed
**Fix**: Verify `quantized_dit.py` exists at `deploy/xvideo/models/quantized_dit.py`

### Issue: "AttributeError: 'Tensor' object has no attribute 'is_quantized'"
**Cause**: Old PyTorch version
**Fix**: Ensure torch >= 1.13 (has quantization support)

### Issue: GPU OOM during model move
**Cause**: Quantized buffers expanded during GPU transfer
**Fix**: Reduce resolution or frame count

### Issue: Output is NaN or garbage
**Cause**: Quantization scale mismatch
**Fix**: Check that quantized checkpoint was created with same quantization scheme

---

## Success Criteria

- [ ] QuantizedTransformer3DModel loads quantized checkpoint without type error
- [ ] Model class correctly identified as QuantizedTransformer3DModel
- [ ] Single forward pass completes without error
- [ ] Output shape and dtype are correct
- [ ] Full inference pipeline generates video
- [ ] Video file is valid and playable
- [ ] GPU memory stays under 20GB

---

## Rollback Plan

If wrapper approach fails:
1. Revert to full-precision checkpoint (larger, but simpler)
2. Or implement streaming inference (load model in parts)

---

## Summary

This approach:
- ✅ Uses quantized checkpoint directly (5.84GB)
- ✅ Avoids type mismatch (buffer-based storage)
- ✅ Avoids 2x memory (no conversion)
- ✅ Preserves quantization throughout
- ✅ Fits within 32GB GPU comfortably (~19GB peak)

**Recommendation**: Test Phase 1-2 immediately to validate approach.
