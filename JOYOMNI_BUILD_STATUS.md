# JoyAI-Video-Edit: joyomni_ops Build Status

**Date:** 2026-08-16  
**Status:** ❌ BLOCKED — Both Windows and horde builds are broken with no fallback option

## Summary

joyomni_ops (CUDA extension for fused DiT operations) fails to build on both target platforms:
- **Windows (local RTX 5090)**: 77 CUDA compilation errors
- **Horde (A40, 48GB)**: Builds successfully but fails at import time

Fallback to PyTorch native implementations is explicitly rejected (no fallback code).

## Windows Build: Failure

**Platform**: Local RTX 5090, 32GB VRAM  
**Toolchain**: MSVC 14.44 (VS 2022 Community), CUDA 12.8, PyTorch 2.9.1 cu130  
**Status**: ❌ 77 compilation errors

### Error Summary

Compilation fails in `fused_norm_scale_shift.cu` with:

1. **PyTorch C++ ABI Mismatch** (5 errors)
   - TensorImpl struct field size mismatches
   - Expected size 4 bytes, got 8 bytes (storage_, autograd_meta_, extra_meta_, version_counter_)
   - Expected total size 160, got 184 (pyobj_slot_ field)
   - Indicates 32-bit vs 64-bit struct layout incompatibility

2. **CUDA Inline Assembly Constraint Errors** (~40 errors)
   - Pointers cannot fit in 32-bit register constraint "r"
   - Affects `__device__` half/bf16 memory operations in CUDA headers
   - Sample: `asm ("ld.global.nc.b32 %0, [%1];" : "=r"(...) : "r"(ptr))`
   - "r" constraint is 32-bit, but pointer is 64-bit

3. **IValue Placement New Errors** (~30 errors)
   - `no instance of overloaded operator new matches argument list`
   - PyTorch's IValue class layout broken with this compiler/version combo
   - Affects `new (&dest) IValue(...)` placement new calls

### Root Cause

Fundamental toolchain incompatibility:
- **PyTorch 2.9.1** expects specific MSVC/CUDA version combinations
- **CUDA 12.8** kernel headers have constraints that don't match 64-bit pointers properly
- **MSVC 14.44** is too strict (or too lenient) about the layout mismatches
- **This codebase was never tested on Windows** with this exact stack

### What We Tried

1. ✅ Set up Developer Command Prompt for MSVC environment
2. ✅ Installed setuptools, wheel, ninja
3. ✅ Fixed CUDA path from v12.4 → v12.8 in setup.py
4. ✅ Verified nvcc is available
5. ❌ Compilation fails before linking even starts

### Files Involved

- `rebuild_joyomni.bat` — Automated build script (sets MSVC env, runs setup.py)
- `deploy/joyomni_ops/setup.py` — Build configuration
- `deploy/joyomni_ops/csrc/fused_norm_scale_shift.cu` — Source file with errors

---

## Horde Build: Import Failure

**Platform**: Horde A40, 48GB VRAM  
**Toolchain**: Linux GCC, CUDA 12.4, PyTorch 2.9.1  
**Status**: ⚠️ Builds successfully, fails at import

### Build Success

The .so file is created:
```
✅ Built: joyomni_ops/_C.cpython-310-x86_64-linux-gnu.so (10 MB)
```

### Import Failure

```
ImportError: dynamic module does not define module export function (PyInit__C)
```

### Root Cause

The codebase uses **TORCH_LIBRARY** pattern (not pybind11):
- Does NOT define `PyInit__C` (pybind11 exports this automatically)
- Registers via `torch.ops.load_library()` instead
- This is the **correct pattern** for torch.ops, but the error message is misleading

### What We Tried

1. ✅ Compiled with CUDA 12.4 environment
2. ✅ Set LD_LIBRARY_PATH for CUDA and PyTorch libs
3. ❌ Import fails looking for PyInit__C (which shouldn't exist)
4. ✅ Verified `torch.ops.load_library()` is the right approach
5. ❌ Still doesn't import properly

### Possible Issues

- `.so` not being located/loaded by torch.ops.load_library()
- .so file is corrupted or incomplete despite successful build
- LD_LIBRARY_PATH not reaching the .so properly
- TORCH_LIBRARY registration not happening

### Files Involved

- `rebuild_joyomni.sh` — Automated build script for Linux
- `deploy/joyomni_ops/__init__.py` — Uses torch.ops.load_library() to register
- `deploy/joyomni_ops/setup.py` — Build configuration

---

## Inference Without joyomni_ops

### Option: PyTorch Fallback (REJECTED)

Suggested fallback approach:
- Replace fused CUDA kernels with native PyTorch operations
- Slower but functional on any CUDA GPU
- Would allow inference to run on both Windows and horde

**User Decision**: ❌ **Explicitly rejected.** No fallback implementations allowed.

### Current Impact

- ❌ Cannot run inference on local Windows RTX 5090
- ❌ Cannot run inference on horde A40
- ✅ Inference code exists ([run_inference_lowmem.py](C:\workspace\world\JoyAI-Video-Edit\run_inference_lowmem.py))
- ✅ Float16 quantization implemented (14GB model on 32GB GPU)
- ✅ Aspect ratio preservation with letterboxing implemented
- ❌ Requires joyomni_ops to proceed

---

## Inference Code Status

### run_inference_lowmem.py
- ✅ Loads DiT in float16 (memory efficient)
- ✅ Handles aspect ratio preservation (letterboxing)
- ✅ Accepts style/reference frame (--ref-image)
- ✅ Reduced resolution (256x256) and frames (1) for memory efficiency
- ⚠️ Depends on joyomni_ops being available

### run_inference.sh
- ✅ Sets up PYTHONPATH and LD_LIBRARY_PATH for joyomni_ops
- ✅ Verifies joyomni_ops import before running
- ✅ Accepts command-line parameters (video, output, frames, resolution, steps)
- ⚠️ Fails at verification step (joyomni_ops not importable)

---

## Path Forward

1. **Fix Windows Build**
   - Try PyTorch 2.8.x or 2.10.x (different ABI/CUDA versions)
   - Rebuild requires full Developer Command Prompt environment
   - May resolve TensorImpl layout mismatches

2. **Fix Horde Import**
   - Debug why torch.ops.load_library() isn't finding/registering .so
   - Check if .so is actually being created with correct symbols
   - Verify LD_LIBRARY_PATH reaches both CUDA and PyTorch libs

3. **Alternative Approaches**
   - Investigate if joyomni_ops is actually required (performance vs functionality)
   - Check if there's a prebuilt .so available for either platform
   - Consider simpler CUDA extension that doesn't hit these ABI issues

---

## Test Commands

### Windows (Developer Command Prompt)
```cmd
cd C:\workspace\world\JoyAI-Video-Edit\deploy\joyomni_ops
C:\workspace\world\JoyAI-Video-Edit\rebuild_joyomni.bat
```

### Horde (Linux)
```bash
cd /workspace/world/JoyAI-Video-Edit
bash rebuild_joyomni.sh
```

### Verify Import (both platforms)
```bash
python -c "from joyomni_ops import fused_norm_scale_shift; print('OK')"
```

### Run Inference (both platforms)
```bash
bash run_inference.sh
```

---

## Files Modified

- [run_inference_lowmem.py](C:\workspace\world\JoyAI-Video-Edit\run_inference_lowmem.py) — Memory-efficient inference script
- [run_inference.sh](C:\workspace\world\JoyAI-Video-Edit\run_inference.sh) — Shell wrapper with joyomni_ops verification
- [rebuild_joyomni.bat](C:\workspace\world\JoyAI-Video-Edit\rebuild_joyomni.bat) — Windows build automation
- [rebuild_joyomni.sh](C:\workspace\world\JoyAI-Video-Edit\rebuild_joyomni.sh) — Linux build automation
- [BUILD_WINDOWS.md](C:\workspace\world\JoyAI-Video-Edit\BUILD_WINDOWS.md) — Windows build guide
- [BUILD_JOYOMNI_OPS.md](C:\workspace\world\JoyAI-Video-Edit\BUILD_JOYOMNI_OPS.md) — General build guide
- [SETUP_AND_RUN.md](C:\workspace\world\JoyAI-Video-Edit\SETUP_AND_RUN.md) — Setup instructions
- [RUN_WITH_JOYOMNI.md](C:\workspace\world\JoyAI-Video-Edit\RUN_WITH_JOYOMNI.md) — Runtime guide

---

## Decision Points

**User Requirements**:
- ✅ Fast inference on local RTX 5090
- ✅ Fast inference on horde A40 (48GB)
- ✅ Use joyomni_ops CUDA extension (no fallback)
- ❌ No PyTorch fallback implementations allowed

**Current Status**: Blocked on joyomni_ops availability.

