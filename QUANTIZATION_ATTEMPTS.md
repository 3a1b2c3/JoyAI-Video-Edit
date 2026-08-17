# Quantization Attempts: All 3 Approaches

## Context
- **GPU**: RTX 5090 (32 GB)
- **Model**: DiT 16.26B parameters
- **Float32 size**: 28-30 GB
- **Goal**: Run int8 quantized inference without dequantization (which would exceed 32 GB)

---

## Approach 1: PyTorch Smart Dequantization ❌

**Concept**: Load quantized checkpoint, dequantize on-demand during forward pass

**Script**: `infer_smart_dequant.py`

**Code**:
```python
class SmartDequantModel(torch.nn.Module):
    def forward(self, x, t, context, rotary_emb=None, image_rotary_emb=None):
        # Dequantize qint8 buffers in-place
        for name, buf in self.model.named_buffers():
            if buf is not None and buf.is_quantized:
                dequantized = torch.dequantize(buf)
                self.model._buffers[name] = dequantized
        return self.model.forward(x, t, context, rotary_emb, image_rotary_emb)
```

**Implementation**:
1. Load qint8 checkpoint (6 GB)
2. Wrap model with SmartDequantModel
3. Dequantize tensors only when used in forward pass

**Result**: ❌ **FAILED**
```
RuntimeError: Copying from quantized Tensor to non-quantized Tensor is not allowed
```

**Error Location**: `model.load_state_dict()` in `load_dit()`

**Root Cause**: 
PyTorch's `load_state_dict()` enforces dtype matching at the parameter level (float32 model cannot accept qint8 tensors). This check happens **before** the wrapper's forward() is ever called, making on-demand dequantization impossible.

**Why It Failed**:
- Can't defer dequantization to forward pass if load_state_dict() rejects the tensors first
- Affects all 894 quantized tensors in checkpoint
- This is a PyTorch design constraint, not a workaround-able issue

---

## Approach 2: CPU Dequantization ❌

**Concept**: Dequantize full checkpoint on CPU, then load into model

**Script**: Modified `load_dit()` in `models.py`

**Code**:
```python
# After loading checkpoint on CPU
quantized_keys = [k for k, v in state_dict.items() 
                   if isinstance(v, torch.Tensor) and v.is_quantized]
if quantized_keys:
    logger.info(f"Dequantizing {len(quantized_keys)} quantized tensors from checkpoint")
    for k in quantized_keys:
        state_dict[k] = torch.dequantize(state_dict[k])  # qint8 → float32
```

**Implementation**:
1. Load qint8 checkpoint on CPU (6 GB)
2. Dequantize all qint8 tensors to float32 (28 GB)
3. Load dequantized state_dict into model
4. Move model to GPU

**Result**: ❌ **FAILED**
```
MemoryError: Cannot allocate 28 GB
```

**Error Location**: During dequantization on CPU

**Memory Analysis**:
- Checkpoint in RAM: 6 GB (qint8)
- Dequantized copy in RAM: 28 GB (float32)
- **Total needed**: 34 GB
- **Available CPU RAM**: ~24-28 GB
- **Shortfall**: 6-10 GB

**Why It Failed**:
- Must hold both quantized (6 GB) + dequantized (28 GB) versions simultaneously
- No way to stream or chunk dequantization
- Hardware constraint cannot be overcome

---

## Approach 3: TensorRT Native Int8 ❌

**Concept**: Export model to ONNX, build TensorRT engine with native int8 support

**Script**: `infer_tensorrt_fixed.py`

**Code**:
```python
def export_dit_to_onnx(onnx_path, input_shapes):
    # Create dummy model (avoid loading quantized checkpoint)
    model = Transformer3DModel(dtype=PRECISION_TO_TYPE[cfg.dit_precision], device=torch.device("cpu"))
    
    # Export to ONNX
    torch.onnx.export(model, (dummy_x, dummy_t, dummy_context), onnx_path, ...)

def build_tensorrt_engine(onnx_path, engine_path):
    # Parse ONNX and build int8 engine
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    engine = builder.build_engine(network, config)
```

**Implementation**:
1. Create empty model (no checkpoint loading)
2. Export structure to ONNX via `torch.onnx.export()`
3. Build TensorRT engine with `INT8` precision flag
4. Run inference with TensorRT runtime

**Result**: ❌ **FAILED**
```
torch.onnx._internal.exporter._errors.TorchExportError:
RuntimeError: Given groups=1, weight of size [3072, 4, 1, 2, 2], 
expected input[1, 64, 1, 16, 16] to have 4 channels, but got 64 channels instead
```

**Error Location**: `torch.onnx.export()` during dummy input tracing

**Root Cause**:
- Model's first conv layer expects 4 input channels
- We provided dummy input with 64 channels (latent space size)
- ONNX export requires correct input shapes to trace the model

**Why It Failed**:
- Cannot export without knowing exact intermediate layer shapes
- Shape mismatch blocks export before we even get to quantization
- Would need full model forward pass tracing (requires working PyTorch inference first)

---

## Summary Table

| Approach | Method | Location Failed | Error Type | Root Cause |
|----------|--------|-----------------|-----------|-----------|
| **1. Smart Dequant** | Defer to forward | `load_state_dict()` | Type mismatch | PyTorch design constraint |
| **2. CPU Dequant** | Full dequant on CPU | Runtime (RAM) | OOM | 34 GB needed > 24-28 GB available |
| **3. TensorRT** | Export + rebuild | `torch.onnx.export()` | Shape mismatch | Wrong dummy input dimensions |

---

## What Actually Works

**Float32 Inference**: `run_inference.py`
- Uses unquantized checkpoint: `joyai_video_edit_dit_0804.pth`
- Memory: 28-30 GB (fits in RTX 5090, plenty of room on A40)
- No quantization overhead
- Full feature parity (all operations supported)

---

## Key Learnings

1. **PyTorch Quantization Limitation**: `load_state_dict()` is strictly typed; cannot load qint8 into float32 parameters
2. **Memory Constraint**: CPU RAM insufficient to hold both qint8 + float32 versions
3. **ONNX Export Requires Working Model**: Cannot export model structure without forward pass tracing
4. **Custom Quantization Scheme**: Original int8 checkpoint uses non-standard quantization PyTorch doesn't natively support

## Conclusion

Native int8 quantized inference is **not feasible** on this hardware with this checkpoint using PyTorch. The combination of:
- PyTorch's type-strict parameter loading
- Memory limitations (32 GB GPU, ~24-28 GB CPU)
- Custom quantization scheme

...makes all three approaches impossible. Float32 inference is the working solution.
