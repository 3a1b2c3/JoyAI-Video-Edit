# JoyAI-Video-Edit: Full-Precision (fp32) Inference Plan

## Decision Context

**Problem**: Quantized checkpoint (qint8) cannot be loaded into float32 model due to PyTorch type restrictions. Dequantization requires 2x memory (both qint8 + float32 copies simultaneously), exceeding 32GB GPU.

**Solution**: Use full-precision (fp32) checkpoint instead. Fits in 32GB GPU with minimal margin. Eliminates type mismatch blocking entirely.

**Trade-off**: Larger model on disk (32.5GB vs 5.84GB qint8), but enables working inference today.

---

## Checkpoint Strategy

| Checkpoint | Size | Location | dtype | Status |
|-----------|------|----------|-------|--------|
| Quantized (qint8) | 5.84 GB | `joyai_video_edit_dit_0804_int8_0815.pth` | qint8 | ❌ Type mismatch |
| **Full-Precision** | **32.5 GB** | **`joyai_video_edit_dit_0804.pth`** | **float32** | **✅ Use this** |

---

## Execution Plan

### Phase 1: Update Inference Scripts (15 min)

**Goal**: Point inference at full-precision checkpoint

#### Step 1.1: Update `joyai_quantized.py`

Replace checkpoint path:
```python
# OLD (quantized, blocked)
DIT_PATH = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804_int8_0815.pth")

# NEW (full-precision)
DIT_PATH = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth")
```

#### Step 1.2: Update `models.py` (cleanup)

Revert to standard loading (no quantization handling needed):
- ✓ Already done (dequantization code removed)
- Keep CPU instantiation + GPU move pattern (prevents OOM during layer creation)

#### Step 1.3: Test single frame encode/decode

```bash
python -c "
import sys; sys.path.insert(0, 'deploy')
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig
import torch

cfg = ExpConfig()
cfg.dit_ckpt = 'deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth'
device = torch.device('cuda')

print('Loading full-precision DiT...')
model = load_dit(cfg, device)
print(f'✓ Loaded: {next(model.parameters()).dtype} on {next(model.parameters()).device}')
"
```

---

### Phase 2: First Inference Run (variable time)

**Goal**: Generate working video output

#### Step 2.1: Test with minimal parameters

```bash
python joyai_quantized.py \
  --frames 2 \
  --steps 2 \
  --height 256 \
  --width 256 \
  --out outputs/test_fp32.mp4
```

Monitor:
- Model load time
- Peak GPU memory
- Frame encode/decode success
- Video save

#### Step 2.2: Monitor GPU memory during inference

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

Expected:
- Model alone: ~28-30 GB
- With VAE + inference: ~31-32 GB (tight but should fit)

#### Step 2.3: If OOM occurs

Options (in priority order):
1. Reduce frame dimensions (256x256 instead of 512x512)
2. Process fewer frames (1-2 instead of 4-8)
3. Enable gradient checkpointing in VAE
4. Use float16 instead of float32 (half memory, halve precision)

---

### Phase 3: Scale to Full Resolution (optional)

Only if Phase 2 succeeds and metrics are acceptable.

```bash
python joyai_quantized.py \
  --frames 8 \
  --steps 4 \
  --height 512 \
  --width 512 \
  --out outputs/full_fp32.mp4
```

---

## Expected Outcomes

| Metric | Quantized | Full-Precision | Notes |
|--------|-----------|-----------------|-------|
| Checkpoint size | 5.84 GB | 32.5 GB | On-disk only |
| GPU memory peak | OOM ❌ | ~32 GB ✓ | Model + inference |
| Inference speed | N/A | ~1-2 fps | Estimate |
| Video quality | Higher | Baseline | fp32 = baseline quality |
| Type mismatch | ❌ Blocked | ✅ Resolved | Core issue solved |

---

## Memory Budget

**GPU: 32 GB total**

```
Model layers (float32):     ~28 GB
VAE (fp16):                ~2 GB
Inference buffers:         ~1.5 GB
PyTorch overhead:          ~0.5 GB
                           ------
Total:                     ~32 GB (tight)
```

If OOM: Reduce resolution or frames immediately.

---

## Files to Modify

| File | Change | Reason |
|------|--------|--------|
| `joyai_quantized.py` | Update DIT_PATH to fp32 | Point to working checkpoint |
| `models.py` | ✓ Already CPU-first pattern | Prevent layer creation OOM |
| README_SETUP.md | Update checkpoint note | Document strategy change |

---

## Rollback Plan

If full-precision doesn't work:
1. Revert `joyai_quantized.py` DIT_PATH back to qint8
2. Revisit Option 2 (quantized wrapper model) with more time/resources

---

## Success Criteria

- [ ] `load_dit()` loads fp32 checkpoint without error
- [ ] Model dtype is float32 (or bfloat16 if configured)
- [ ] Model on CUDA without OOM
- [ ] Single frame encode/decode completes
- [ ] Full 2-frame inference runs to completion
- [ ] Video saved to outputs/ without corruption
- [ ] GPU memory peak < 32 GB

---

## Timeline

- **Phase 1 (Update scripts)**: 15 min
- **Phase 2 (First inference)**: 10-30 min (depending on speed)
- **Phase 3 (Scale)**: 5-10 min per run

**Total**: ~1 hour to first video output

---

## Notes

- Full-precision model was already quantized once; re-quantizing on-the-fly won't recover the original speed benefit
- This is a pragmatic unblock, not the long-term solution
- If real quantized inference needed later, must revisit Option 2 (wrapper model) with allocated development time
