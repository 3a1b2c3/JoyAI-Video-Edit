#!/bin/bash
# Run DiT inference on horde with joyomni_ops

set -euo pipefail  # fail on error, undefined vars, pipe failures

trap 'echo "ERROR on line $LINENO"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""
echo "DEBUG: System Info"
$PYTHON -c "import sys, torch; print(f'  Python: {sys.version}'); print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
echo ""

# Setup PYTHONPATH for joyomni_ops (MUST be first for import to work)
export PYTHONPATH="$SCRIPT_DIR/deploy/joyomni_ops:$SCRIPT_DIR/deploy:${PYTHONPATH:-}"

# Setup LD_LIBRARY_PATH for CUDA and PyTorch
TORCH_LIB=$($PYTHON -c "import torch; print(torch.__path__[0])")/lib
CUDA_LIB="${CUDA_HOME:-/usr/local/cuda-12.4}/lib64"
export LD_LIBRARY_PATH="$CUDA_LIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# Setup checkpoint caching (faster loading on horde)
export TORCH_HOME="$SCRIPT_DIR/.cache/torch"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

# Fix GPU memory fragmentation on 48GB systems near capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $CUDA_LIB:$TORCH_LIB"
echo "  TORCH_HOME: $TORCH_HOME (cached)"
echo "  HF_HOME: $HF_HOME (cached)"
echo ""

# Parse arguments
VIDEO="${1:-assets/cases/omnidream/mattress.mp4}"
OUTPUT="${2:-$SCRIPT_DIR/outputs/dit_output.mp4}"
REF_IMAGE="${3:-assets/image.png}"
FRAMES="${4:-1}"
HEIGHT="${5:-256}"
WIDTH="${6:-256}"
STEPS="${7:-1}"

echo "Configuration:"
echo "  Video:      $VIDEO"
echo "  Output:     $OUTPUT"
echo "  Style:      $REF_IMAGE"
echo "  Frames:     $FRAMES"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo "  Steps:      $STEPS"
echo ""

# Verify input video exists
if [ ! -f "$VIDEO" ]; then
    echo "❌ ERROR: Input video not found: $VIDEO"
    exit 1
fi
echo "✓ Input video found"

# Verify style image if specified
if [ -n "$REF_IMAGE" ] && [ "$REF_IMAGE" != "" ]; then
    if [ ! -f "$REF_IMAGE" ]; then
        echo "⚠ WARNING: Style image not found: $REF_IMAGE (proceeding without style)"
        REF_IMAGE=""
    else
        echo "✓ Style image found"
    fi
fi

# Verify joyomni_ops (fail hard if missing)
echo "Checking joyomni_ops..."
if ! OUTPUT=$($PYTHON -c "from joyomni_ops import fused_norm_scale_shift; print('✅ joyomni_ops available')" 2>&1); then
    echo ""
    echo "=========================================="
    echo "❌ FATAL: joyomni_ops import failed"
    echo "=========================================="
    echo ""
    echo "Error output:"
    echo "$OUTPUT"
    echo ""
    echo "joyomni_ops._C.cpython-310-x86_64.pyd must be built first."
    echo ""
    echo "Check if .pyd exists:"
    echo "  ls -la deploy/joyomni_ops/joyomni_ops/_C*.pyd"
    echo ""
    echo "If missing, rebuild:"
    echo "  cd deploy/joyomni_ops"
    echo "  set JOYOMNI_OPS_NO_FP8=1"
    echo "  python setup.py build_ext --inplace"
    echo ""
    exit 1
fi
echo "$OUTPUT"
echo ""

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Cannot create output directory: $OUTPUT_DIR"
    exit 1
fi
echo "✓ Output directory ready: $OUTPUT_DIR"
echo ""

# Run inference
echo "Running inference..."
echo "  Frames: $FRAMES"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo "  Steps: $STEPS"
echo ""

# Load models and run diffusion
$PYTHON << 'PYEOF'
import sys
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import inference components
sys.path.insert(0, './deploy')
from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.models.vae import XVAEChunkCausal
from xvideo.config import ExpConfig
from xvideo.models.pipeline import PRECISION_TO_TYPE

device = torch.device("cuda")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
print(f"VRAM Reserved: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
print()

seed_everything(42)

# Load video
video_path = sys.argv[1] if len(sys.argv) > 1 else "assets/Recording 2026-08-12 205529.mp4"
frames_to_load = int(sys.argv[4]) if len(sys.argv) > 4 else 1
height = int(sys.argv[5]) if len(sys.argv) > 5 else 256
width = int(sys.argv[6]) if len(sys.argv) > 6 else 256

print(f"[1/5] Loading video: {video_path}")
if not Path(video_path).exists():
    print(f"ERROR: Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

for i in range(min(frames_to_load, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ok, bgr = cap.read()
    if not ok:
        break
    # Letterbox with aspect ratio preservation
    orig_h, orig_w = bgr.shape[:2]
    aspect = orig_w / orig_h
    if aspect > width / height:
        new_w, new_h = width, int(width / aspect)
    else:
        new_h, new_w = height, int(height * aspect)
    bgr_resized = cv2.resize(bgr, (new_w, new_h))
    pad_t, pad_b = (height - new_h) // 2, height - new_h - (height - new_h) // 2
    pad_l, pad_r = (width - new_w) // 2, width - new_w - (width - new_w) // 2
    bgr_padded = cv2.copyMakeBorder(bgr_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
    rgb = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2RGB)
    frames.append(torch.from_numpy(rgb).float() / 255.0)
cap.release()

frames_tensor = torch.stack(frames).to(device, dtype=torch.bfloat16)
print(f"✓ Loaded {len(frames)} frames @ {fps:.1f} fps")
print()

# Load models with memory optimization
print("[2/5] Loading models (float16, memory-efficient)...")
import gc

cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"  # joyomni_ops CUDA kernels require bf16, not fp16
cfg.dit_ckpt = str(Path("deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth"))

# Load DiT directly in float16 (avoiding temporary float32 copy that causes OOM)
print("  Loading DiT (direct to GPU in float16, no conversion)...")
mem_before = torch.cuda.memory_allocated() / 1e9
print(f"    GPU memory before: {mem_before:.1f}GB")

dit = load_dit(cfg, device=device)
mem_after_load = torch.cuda.memory_allocated() / 1e9
print(f"    GPU memory after load: {mem_after_load:.1f}GB (+{mem_after_load - mem_before:.1f}GB)")

# Verify dtype
model_dtype = next(dit.parameters()).dtype
print(f"    Model dtype: {model_dtype}")

dit.eval()
dit.requires_grad_(False)
torch.cuda.empty_cache()
mem_after_cleanup = torch.cuda.memory_allocated() / 1e9
print(f"  ✓ DiT loaded (final: {mem_after_cleanup:.1f}GB)")

# Load VAE + text encoder on CPU (DiT uses all GPU memory)
print("  Loading VAE (float16, CPU)...")
vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.float16)
vae = vae.to("cpu")
vae.eval()
vae.requires_grad_(False)
print(f"  ✓ VAE loaded (float16, CPU)")

# Load text encoder on CPU if available
try:
    print("  Loading text encoder (CPU)...")
    text_encoder_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder")
    if text_encoder_ckpt.exists():
        from xvideo.models.models import load_text_encoder
        tokenizer, text_encoder = load_text_encoder(
            str(text_encoder_ckpt),
            device=torch.device("cpu"),
            torch_dtype=torch.float16
        )
        text_encoder.eval()
        print(f"  ✓ Text encoder loaded (CPU)")
except Exception as e:
    print(f"  ⚠ Text encoder not available: {e}")
    text_encoder = None
    tokenizer = None

print(f"  Note: VAE/text encode/decode run on CPU (slower but fits GPU)")

# Clear memory and summary
gc.collect()
torch.cuda.empty_cache()
mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
mem_allocated = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
print(f"  GPU memory summary:")
print(f"    Allocated: {mem_allocated:.1f}GB")
print(f"    Reserved:  {mem_reserved:.1f}GB")
print(f"    Total:     {mem_total:.1f}GB")
print(f"    Available: {mem_total - mem_reserved:.1f}GB")
print()

# Encode
print("[3/5] VAE encoding...")
mem_before_encode = torch.cuda.memory_allocated() / 1e9
print(f"  Memory before encoding: {mem_before_encode:.1f}GB")
with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2)
    # Move to CPU for VAE encoding (VAE is on CPU)
    frames_chw = frames_chw.to("cpu")
    latents_list = []
    for i in tqdm(range(len(frames_chw)), desc="Encoding"):
        try:
            z = frames_chw[i:i+1].unsqueeze(2)
            print(f"    Frame {i}: input shape {z.shape}, device {z.device}")
            posterior = vae.encode(z).latent_dist
            sample = posterior.sample() * 0.18215
            print(f"    Frame {i}: latent shape {sample.shape}, channels={sample.shape[1]}, device {sample.device}")
            if sample.shape[1] != 64:
                print(f"    ⚠ WARNING: Expected 64 channels, got {sample.shape[1]}")
            latents_list.append(sample)
        except Exception as e:
            print(f"    Frame {i} ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    print(f"  latents_list length: {len(latents_list)}")
    if latents_list:
        latents = torch.cat(latents_list, dim=0)
    else:
        print("  WARNING: latents_list empty, using raw frames")
        latents = frames_chw[:1]
    # Move latents back to GPU for diffusion
    latents = latents.to("cuda")
    print(f"  Final latents shape: {latents.shape}")
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_encode = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Encoded to {latents.shape}")
    print(f"  Memory: {mem_after_encode:.1f}GB ({mem_after_encode - mem_before_encode:+.1f}GB)")
print()

# Diffusion
steps = int(sys.argv[7]) if len(sys.argv) > 7 else 1
print(f"[4/5] Diffusion ({steps} steps)...")
mem_before_diffusion = torch.cuda.memory_allocated() / 1e9
print(f"  Memory before diffusion: {mem_before_diffusion:.1f}GB")
with torch.no_grad():
    for step in tqdm(range(steps), desc="Denoising"):
        t = (steps - step - 1) / steps
        t_idx = int(t * 1000)
        t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)
        context = torch.randn(latents.shape[0], 256, 4096, dtype=torch.bfloat16, device=device)
        model_output = dit(latents, t_tensor, context)
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]
        sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
        latents = latents + model_output * (-sigma * 0.1)
        torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
mem_after_diffusion = torch.cuda.memory_allocated() / 1e9
print(f"✓ Diffusion complete")
print(f"  Memory: {mem_after_diffusion:.1f}GB ({mem_after_diffusion - mem_before_diffusion:+.1f}GB)")
print()

# Decode
print("[5/5] Decoding...")
mem_before_decode = torch.cuda.memory_allocated() / 1e9
print(f"  Memory before decoding: {mem_before_decode:.1f}GB")
with torch.no_grad():
    latents_decoded = latents / 0.18215
    frames_decoded = []
    for i in tqdm(range(len(latents_decoded)), desc="Decoding"):
        try:
            frame = vae.decode(latents_decoded[i:i+1]).sample
            frames_decoded.append(frame)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Frame {i}: {e}")
            continue
    decoded = torch.cat(frames_decoded, dim=0) if frames_decoded else latents_decoded[:1]
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_decode = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Decoded {len(decoded)} frames")
    print(f"  Memory: {mem_after_decode:.1f}GB ({mem_after_decode - mem_before_decode:+.1f}GB)")

# Save
output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/dit_output.mp4"
output_path = str(Path(output_path).resolve())  # absolute path (cwd = script dir)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

import imageio.v3 as iio
if decoded.ndim == 5:
    decoded = decoded.squeeze(2)
output_frames = (decoded.to(torch.float32).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
iio.imwrite(output_path, output_frames, fps=fps)
print(f"\n✓ Saved: {output_path}")
print("=" * 70)
print("✅ INFERENCE COMPLETE")
print("=" * 70)

PYEOF

PYEOF_EXIT=$?
if [ $PYEOF_EXIT -ne 0 ]; then
    echo "Inference failed"
    exit 1
fi

echo ""
echo "✅ Done!"
echo "Output: $OUTPUT"
