# JoyAI-Video-Edit Standalone Inference Status

## Current State
✅ **Working:**
- Models load successfully (42.6GB on A40)
- GPU auto-detection & optimization
- Triton kernel autotuning & caching
- Environment setup & wrappers

❌ **Blocker: Inference API Unknown**

## The Problem
The standalone inference script can't call the Pipeline to generate output.

**Root cause:** Pipeline object is not directly callable.
- `pipeline(...)` → `TypeError: 'Pipeline' object is not callable`
- `pipeline.__call__(...)` → `AttributeError: no attribute '__call__'`
- `runtime.pipeline(...)` → Same issue

## What We Know
- `JoyOmniRuntime.load()` works and returns a runtime object
- `runtime.pipeline` exists (line 199 of joyomni_streaming.py)
- Pipeline extends DiffusionPipeline but doesn't implement standard interface
- The server uses JoyOmniRuntime for streaming, not one-shot inference

## Next Steps to Fix
1. Check `run_server.sh` to see how it actually invokes inference
2. Examine `joyomni_streaming.py` for streaming inference methods
3. Find if Pipeline has a non-standard inference method name
4. OR use the server's streaming/chunking interface adapted for single-frame inference

## Test Scripts
- `infer_standalone.sh` — wrapper with GPU auto-detection
- `precompile_kernels.sh` — warmup for kernel caching
- `deploy/infer_standalone.py` — inference script (needs API fix)

## Checkpoint Files
All downloaded and verified:
- DiT: 31GB
- VAE: present
- Text encoder (MiMo): present
