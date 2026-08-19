#!/bin/bash
# Run DiT inference on horde with joyomni_ops

set -u  # fail on undefined vars (but not on errors or pipe failures)

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

# Parse arguments with sensible defaults
VIDEO="${1:-assets/cases/omnidream/mattress.mp4}"
OUTPUT="${2:-$SCRIPT_DIR/outputs/stylized_output.mp4}"
REF_IMAGE="${3:-assets/image.png}"
FRAMES="${4:-10}"  # number of frames (10 = ~0.3 sec @ 30fps), "all" = entire video
HEIGHT="${5:-auto}"
WIDTH="${6:-auto}"
STEPS="${7:-20}"  # More steps for better quality (1-50, default 20)
CFG="${8:-15.0}"  # CFG scale for style guidance (1.0 = none, 15.0 = very strong)
PROMPT="${9:-}"  # Text prompt for guidance (optional, complements style image)

# Resolve full output path upfront
OUTPUT_FULL=$(cd "$(dirname "$SCRIPT_DIR/$OUTPUT")" 2>/dev/null && pwd -P)/$(basename "$OUTPUT") || echo "$SCRIPT_DIR/$OUTPUT"

echo "Configuration:"
echo "  Video:      $VIDEO"
echo "  Output:     $OUTPUT_FULL"
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

# Export args as env vars for Python
export JOYAI_VIDEO="$VIDEO"
export JOYAI_OUTPUT="$OUTPUT"
export JOYAI_REF_IMAGE="$REF_IMAGE"
export JOYAI_FRAMES="$FRAMES"
export JOYAI_HEIGHT="$HEIGHT"
export JOYAI_WIDTH="$WIDTH"
export JOYAI_STEPS="$STEPS"
export JOYAI_CFG="$CFG"
export JOYAI_PROMPT="$PROMPT"

# Load models and run diffusion
$PYTHON -u << 'PYEOF'
import os
import sys
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Memory debug function
def mem_info(label=""):
    if label:
        print(f"[MEM] {label}", flush=True)
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    available = total - reserved
    print(f"  Allocated: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {total:.1f}GB / Free: {available:.1f}GB", flush=True)

# Ensure errors are always printed
import traceback as tb_module
def show_error(exc_type, exc_value, exc_traceback):
    print(f"\n{'='*70}")
    print(f"❌ UNHANDLED ERROR: {exc_type.__name__}")
    print(f"{'='*70}")
    tb_module.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)
sys.excepthook = show_error

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
mem_info("Start")
print()

seed_everything(42)

# Load video
video_path = os.environ.get("JOYAI_VIDEO", "assets/Recording 2026-08-12 205529.mp4")
frames_arg = os.environ.get("JOYAI_FRAMES", "10")
# H/W: explicit digits are used (letterboxed to that box); otherwise AUTO -> match
# the video's aspect ratio (longest side = BASE, snapped to a multiple of 16 for the VAE).
_h_arg = sys.argv[5] if len(sys.argv) > 5 else ""
_w_arg = sys.argv[6] if len(sys.argv) > 6 else ""
height = int(_h_arg) if _h_arg.isdigit() else None
width = int(_w_arg) if _w_arg.isdigit() else None
BASE = 256

print(f"[1/5] Loading video: {video_path}")
if not Path(video_path).exists():
    print(f"ERROR: Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_to_load = total_frames if frames_arg == "all" else int(frames_arg)

# Auto-size to the source aspect ratio when H/W aren't given explicitly.
auto_size = height is None or width is None
if auto_size:
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 16
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 16
    ar = src_w / src_h
    _snap = lambda x: max(16, int(round(x / 16)) * 16)
    if ar >= 1:   # landscape
        width, height = BASE, _snap(BASE / ar)
    else:         # portrait
        height, width = BASE, _snap(BASE * ar)
    print(f"  Auto resolution from {src_w}x{src_h} (AR {ar:.3f}) -> {width}x{height}")

frames = []
for i in range(min(frames_to_load, total_frames)):
    ok, bgr = cap.read()
    if not ok:
        break
    if auto_size:
        # aspect already matched -> plain resize, no letterbox padding
        bgr_out = cv2.resize(bgr, (width, height))
    else:
        # explicit H/W -> letterbox to preserve aspect inside that box
        orig_h, orig_w = bgr.shape[:2]
        aspect = orig_w / orig_h
        if aspect > width / height:
            new_w, new_h = width, int(width / aspect)
        else:
            new_h, new_w = height, int(height * aspect)
        bgr_resized = cv2.resize(bgr, (new_w, new_h))
        pad_t, pad_b = (height - new_h) // 2, height - new_h - (height - new_h) // 2
        pad_l, pad_r = (width - new_w) // 2, width - new_w - (width - new_w) // 2
        bgr_out = cv2.copyMakeBorder(bgr_resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    frames.append(torch.from_numpy(rgb).float() / 255.0)
cap.release()

frames_tensor = torch.stack(frames).to(device, dtype=torch.bfloat16)
print(f"✓ Loaded {len(frames)} frames @ {fps:.1f} fps")
print()

# Load models with memory optimization
print("[2/5] Loading models (float16, memory-efficient)...")
import gc

# Verify quantized checkpoint exists
dit_ckpt_path = Path("dit_quantized.pth")
if not dit_ckpt_path.exists():
    print(f"ERROR: Quantized checkpoint not found: {dit_ckpt_path}")
    print("Run: python quantize_simple.py <input> dit_quantized.pth")
    sys.exit(1)

cfg = ExpConfig()
cfg.training_mode = False
cfg.dit_precision = "bf16"  # joyomni_ops CUDA kernels require bf16, not fp16
cfg.dit_ckpt = str(dit_ckpt_path)  # Use quantized (2x smaller) checkpoint

# Load DiT with quantized checkpoint (auto-dequantizes int8 tensors)
print("  Loading DiT (quantized checkpoint, auto-dequantized to bf16)...")
mem_before_load = torch.cuda.memory_allocated() / 1e9
mem_info("Before load_dit()")

dit = load_dit(cfg, device=device)
mem_info("After load_dit()")
mem_after_load = torch.cuda.memory_allocated() / 1e9
print(f"    GPU memory after load: {mem_after_load:.1f}GB (+{mem_after_load - mem_before_load:.1f}GB)")

# Verify dtype
model_dtype = next(dit.parameters()).dtype
print(f"    Model dtype: {model_dtype}")

dit.eval()
dit.requires_grad_(False)
torch.cuda.empty_cache()
mem_after_cleanup = torch.cuda.memory_allocated() / 1e9
print(f"  ✓ DiT loaded (final: {mem_after_cleanup:.1f}GB)")

# Load VAE in bf16 (matches DiT dtype)
print("  Loading VAE (bf16, CPU)...")
vae_ckpt = Path("deploy/deps/checkpoints/JoyAI-Video-Edit/vae")
vae = XVAEChunkCausal.from_pretrained(str(vae_ckpt), torch_dtype=torch.bfloat16)
vae = vae.to("cpu")
# Force all VAE parameters/buffers to bf16 (fixes bias dtype mismatches)
for param in vae.parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)
for buf in vae.buffers():
    if buf.dtype not in (torch.long, torch.int, torch.bool):
        if buf.dtype != torch.bfloat16:
            buf.data = buf.data.to(torch.bfloat16)
vae.eval()
vae.requires_grad_(False)
print(f"  ✓ VAE loaded (bf16, CPU)")
mem_info("After VAE load")

# Load Qwen2.5-VL for image encoding
try:
    print("  Loading Qwen2.5-VL for image encoding...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        local_files_only=False,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    qwen_model.eval()
    print(f"  ✓ Qwen2.5-VL loaded (CPU)")
except Exception as e:
    print(f"  ⚠ Qwen encoder not available: {e}")
    qwen_processor = None
    qwen_model = None

print(f"  Note: DiT 32GB on GPU + VAE 3GB on CPU + Qwen2.5-VL on CPU")

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
print(f"  [DEBUG] frames_tensor: {frames_tensor.shape} {frames_tensor.dtype} on {frames_tensor.device}")

with torch.no_grad():
    frames_chw = frames_tensor.permute(0, 3, 1, 2)
    print(f"  [DEBUG] After permute: {frames_chw.shape}")
    frames_chw = frames_chw.to("cpu")
    print(f"  [DEBUG] Moved to CPU: {frames_chw.device}")

    latents_list = []
    for i in tqdm(range(len(frames_chw)), desc="Encoding"):
        try:
            z = frames_chw[i:i+1].unsqueeze(2)
            print(f"  [DEBUG] Frame {i}: z shape={z.shape} on {z.device}")
            posterior = vae.encode(z).latent_dist
            sample = posterior.sample() * 0.18215
            print(f"  [DEBUG] Frame {i}: latent shape={sample.shape}")
            latents_list.append(sample)
        except Exception as e:
            print(f"  ⚠ Frame {i} encode error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"  [DEBUG] Encoded {len(latents_list)} frames successfully")
    if latents_list:
        latents = torch.cat(latents_list, dim=0)
        print(f"  [DEBUG] Concatenated latents: {latents.shape}")
    else:
        print("  WARNING: latents_list empty, using raw frames")
        latents = frames_chw[:1]

    # Move latents to GPU for diffusion
    print(f"  [DEBUG] Moving latents to GPU...")
    latents = latents.to("cuda")
    print(f"  [DEBUG] Latents on GPU: {latents.shape} {latents.dtype} on {latents.device}")

    gc.collect()
    torch.cuda.empty_cache()
    mem_after_encode = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Encoded {len(latents)} frames")
    print(f"  Memory: {mem_after_encode:.1f}GB ({mem_after_encode - mem_before_encode:+.1f}GB)")
print()

# Encode style image using pipeline.py approach
context_style = None
style_image_path = os.environ.get("JOYAI_REF_IMAGE", "/home/horde/JoyAI-Video-Edit/assets/image.png")

# VALIDATION: Check if style image exists
if not Path(style_image_path).exists():
    print(f"❌ ERROR: Style image not found: {style_image_path}")
    sys.exit(1)

if qwen_processor is None or qwen_model is None:
    print(f"❌ ERROR: Qwen2.5-VL encoder not available (failed to load)")
    sys.exit(1)

print(f"[3.5/5] Encoding style image with Qwen2.5-VL (pipeline approach)...")

try:
    from PIL import Image
    from xvideo.models.pipeline import Pipeline

    # Load image
    style_img = Image.open(style_image_path).convert("RGB")
    print(f"  ✓ Loaded image: {style_image_path} ({style_img.size})")

    # Encode directly with image token markers (like pipeline.py does)
    prompt_text = "Describe the visual style, colors, atmosphere, and artistic composition of this image.\n<|vision_start|><|image_pad|><|vision_end|>"

    # Process text + image
    inputs = qwen_processor(text=prompt_text, images=[style_img], return_tensors="pt")
    print(f"  [DEBUG] Processor keys: {list(inputs.keys())}")

    # Run model forward
    with torch.no_grad():
        outputs = qwen_model(**{k: v.to("cpu") for k, v in inputs.items()}, output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-1]  # [1, seq_len, 4096]

    print(f"  [DEBUG] Embeddings shape: {prompt_embeds.shape}")
    if prompt_embeds is None or prompt_embeds.shape[0] == 0:
        raise ValueError(f"Encoding returned empty: {prompt_embeds.shape if prompt_embeds is not None else 'None'}")

    # Get embedding dimension
    emb_dim = prompt_embeds.shape[-1]
    print(f"  [DEBUG] Embedding dim: {emb_dim}")

    # Project 3584 → 4096 if needed
    if emb_dim != 4096:
        print(f"  [DEBUG] Projecting {emb_dim} → 4096 (dtype: {prompt_embeds.dtype})")
        proj = torch.nn.Linear(emb_dim, 4096, dtype=prompt_embeds.dtype, device="cpu")
        with torch.no_grad():
            prompt_embeds = proj(prompt_embeds)
        print(f"  [DEBUG] After projection: {prompt_embeds.shape}")

    # Trim/pad to [1, 256, 4096]
    seq_len = prompt_embeds.shape[1]
    if seq_len >= 256:
        context_style = prompt_embeds[:, :256, :]
    else:
        pad_size = 256 - seq_len
        context_style = torch.cat([
            prompt_embeds,
            torch.zeros(1, pad_size, 4096, device=prompt_embeds.device)
        ], dim=1)

    context_style = context_style.to(device).to(torch.bfloat16)
    print(f"  ✓ Context shape: {context_style.shape}, dtype: {context_style.dtype}")

# Repeat context to match latent batch size
if context_style is not None:
    batch_size = latents.shape[0]
    if context_style.shape[0] == 1 and batch_size > 1:
        context_style = context_style.repeat(batch_size, 1, 1)
        print(f"  [DEBUG] Repeated context to batch size {batch_size}: {context_style.shape}")
except Exception as e:
    print(f"❌ ERROR: Style encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Diffusion with Classifier-Free Guidance (CFG) + Style
steps = int(sys.argv[7]) if len(sys.argv) > 7 else 20
cfg_scale = float(os.environ.get("JOYAI_CFG", "7.5"))  # Guidance scale
guidance_mode = "style" if context_style is not None else "random"
print(f"[4/5] Diffusion ({steps} steps, CFG={cfg_scale}, context={guidance_mode})...")
mem_before_diffusion = torch.cuda.memory_allocated() / 1e9
print(f"  Memory before diffusion: {mem_before_diffusion:.1f}GB")

print(f"  [DEBUG] Starting diffusion loop with {steps} steps")
print(f"  [DEBUG] Latents shape: {latents.shape}, device: {latents.device}")
print(f"  [DEBUG] Context shape: {context_style.shape if context_style is not None else 'None'}")

with torch.no_grad():
    for step in tqdm(range(steps), desc="Denoising"):
        t = (steps - step - 1) / steps
        t_idx = int(t * 1000)
        t_tensor = torch.full((latents.shape[0],), t_idx, device=device, dtype=torch.long)

        # Use style-encoded context if available, otherwise random
        if context_style is not None:
            context = context_style
        else:
            context = torch.randn(latents.shape[0], 256, 4096, dtype=torch.bfloat16, device=device)

        if step == 0:
            print(f"  [DEBUG] Step {step}: t={t:.3f}, t_idx={t_idx}, context shape={context.shape}, device={context.device}")

        # Conditional forward pass (with context)
        output_cond = dit(latents, t_tensor, context)
        if isinstance(output_cond, (tuple, list)):
            output_cond = output_cond[0]

        # Unconditional forward pass (empty context for CFG)
        context_uncond = torch.zeros_like(context)
        output_uncond = dit(latents, t_tensor, context_uncond)
        if isinstance(output_uncond, (tuple, list)):
            output_uncond = output_uncond[0]

        # Apply Classifier-Free Guidance
        model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

        if step == 0:
            print(f"  [DEBUG] Step {step}: output shapes: cond={output_cond.shape}, uncond={output_uncond.shape}, final={model_output.shape}")

        sigma = np.sqrt(t / (1 - t)) if t > 0 else 0
        latents = latents + model_output * (-sigma * 0.1)

        if step == 0 or step == steps - 1:
            mem_diffusion = torch.cuda.memory_allocated() / 1e9
            print(f"  [DEBUG] Step {step}: latents updated, memory: {mem_diffusion:.1f}GB")

        torch.cuda.empty_cache()

# Move model back to CPU to free GPU
dit.to("cpu")
gc.collect()
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
    # Move to CPU if VAE is on CPU (device mismatch fix)
    if str(vae.device) == 'cpu':
        latents_decoded = latents_decoded.to("cpu")
    frames_decoded = []
    for i in tqdm(range(len(latents_decoded)), desc="Decoding"):
        try:
            frame = vae.decode(latents_decoded[i:i+1]).sample
            # Ensure output is on GPU if we need it there
            frame = frame.to(device)
            frames_decoded.append(frame)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Frame {i}: {e}")
            import traceback
            traceback.print_exc()
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

try:
    import imageio.v3 as iio

    # Handle different shape formats
    if decoded.ndim == 5:
        decoded = decoded.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
    if decoded.ndim == 4:
        # (B, C, H, W) -> (B, H, W, C) for video
        decoded = decoded.permute(0, 2, 3, 1)

    # Normalize to uint8
    output_frames = (decoded.to(torch.float32) * 127.5 + 128).clamp(0, 255).cpu().numpy().astype(np.uint8)
    print(f"  Saving {len(output_frames)} frames...")
    iio.imwrite(output_path, output_frames, fps=fps)
    print(f"\n✓ Output saved:")
    print(f"  {output_path}")
except Exception as e:
    print(f"\n❌ ERROR saving output: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)
print("=" * 70)
print("✅ INFERENCE COMPLETE")
print("=" * 70)
print(f"Full path: {output_path}")
PYEOF

PYEOF_EXIT=$?
if [ $PYEOF_EXIT -ne 0 ]; then
    echo "Inference failed"
    exit 1
fi

echo ""
echo "✅ Done!"
echo "Output: $OUTPUT"
