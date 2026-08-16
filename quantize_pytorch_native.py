#!/usr/bin/env python3
"""Quantize JoyAI DiT using PyTorch native quantization (dynamic int8)."""

import sys
import argparse
from pathlib import Path
import time

import torch
import torch.quantization as tq
import warnings

# Suppress deprecation warning for torch.quantization (still works fine)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

def quantize_dit_dynamic(model, device="cuda"):
    """
    Apply PyTorch dynamic quantization to DiT.

    Dynamic quantization:
    - No calibration needed
    - Activations quantized at runtime (int8)
    - Weights quantized upfront (int8)
    - 75-85% size reduction typical
    - Minimal accuracy loss for inference
    """
    print("\n[Quantization] Applying dynamic int8 quantization...")
    print("  Method: torch.quantization.quantize_dynamic")
    print("  Qconfig: default (symmetric, per-tensor)")

    # Move to CPU for quantization (quantization ops run on CPU)
    model = model.cpu()

    # Dynamic quantization: quantize weights, activations at runtime
    quantized_model = tq.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8,  # Use int8
    )

    print("✓ Dynamic quantization complete")

    return quantized_model

def quantize_dit_static_calibrated(model, calibration_data=None, device="cuda"):
    """
    Apply PyTorch static quantization (post-training, with calibration).

    Static quantization:
    - Better accuracy than dynamic (uses calibration data)
    - Slower inference (no runtime adaptive quantization)
    - Requires representative calibration data
    - ~80% size reduction
    """
    if calibration_data is None:
        print("⚠️  Static quantization requires calibration data")
        print("   Falling back to dynamic quantization...")
        return quantize_dit_dynamic(model, device)

    print("\n[Quantization] Applying static int8 quantization (calibrated)...")
    print("  Method: torch.quantization.prepare_qat → convert")
    print("  Qconfig: x86 default (symmetric, per-tensor)")

    model = model.cpu()

    # Set quantization config for static quantization
    model.qconfig = tq.get_default_qconfig('x86')

    # Prepare for quantization (insert fake quant layers)
    tq.prepare(model, inplace=True)

    # Calibrate with data (model runs in eval mode, collects statistics)
    print("  Calibrating on representative data...")
    with torch.no_grad():
        for i, calib_batch in enumerate(calibration_data):
            if i >= 10:  # Use first 10 batches for calibration
                break
            model(calib_batch)

    # Convert to actual quantized model
    tq.convert(model, inplace=True)

    print("✓ Static quantization complete")

    return model

def main():
    parser = argparse.ArgumentParser(description="Quantize JoyAI DiT with PyTorch native quantization")
    parser.add_argument("--input-checkpoint",
                        default="deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth",
                        help="Input fp32 checkpoint path")
    parser.add_argument("--output",
                        default="deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811_quantized.pth",
                        help="Output quantized checkpoint path")
    parser.add_argument("--method", choices=["dynamic", "static"], default="dynamic",
                        help="Quantization method (dynamic=faster, static=better accuracy)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device for quantization")
    args = parser.parse_args()

    print("=" * 80)
    print("JoyAI DiT - PyTorch Native Quantization")
    print("=" * 80)

    input_path = Path(args.input_checkpoint)
    output_path = Path(args.output)

    # Validate input
    if not input_path.exists():
        print(f"\n✗ ERROR: Input checkpoint not found: {input_path}")
        return 1

    input_size_gb = input_path.stat().st_size / 1e9
    print(f"\n[Input] Checkpoint: {input_path.name}")
    print(f"  Size: {input_size_gb:.2f} GB")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[Setup] Device: {device}")

    # Load full-precision model
    print(f"\n[1/4] Loading checkpoint...")
    start_load = time.time()
    try:
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(input_path)

        dit = load_dit(cfg, device=device)
        dit.eval()
        dit.requires_grad_(False)

        # Convert to float32 if needed (quantization needs float32)
        current_dtype = next(dit.parameters()).dtype
        if current_dtype != torch.float32:
            print(f"  Converting from {current_dtype} → float32...")
            dit = dit.to(torch.float32)

        load_time = time.time() - start_load
        print(f"✓ Loaded in {load_time:.1f}s (dtype: {next(dit.parameters()).dtype})")
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Count parameters
    param_count = sum(p.numel() for p in dit.parameters())
    print(f"  Parameters: {param_count / 1e9:.2f}B")

    # Quantize
    print(f"\n[2/4] Quantizing ({args.method})...")
    start_quant = time.time()
    try:
        if args.method == "dynamic":
            quantized_dit = quantize_dit_dynamic(dit, device=device)
        else:
            # Static quantization (no calibration data in this example)
            quantized_dit = quantize_dit_static_calibrated(dit, calibration_data=None, device=device)
        quant_time = time.time() - start_quant
        print(f"  Time: {quant_time:.1f}s")
    except Exception as e:
        print(f"✗ ERROR: Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save quantized model
    print(f"\n[3/4] Saving quantized checkpoint...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_save = time.time()
    try:
        torch.save(quantized_dit.state_dict(), output_path)
        save_time = time.time() - start_save
        output_size_gb = output_path.stat().st_size / 1e9
        compression_ratio = input_size_gb / output_size_gb
        print(f"✓ Saved: {output_path.name}")
        print(f"  Size: {output_size_gb:.2f} GB")
        print(f"  Compression: {compression_ratio:.1f}× smaller ({(1 - 1/compression_ratio) * 100:.1f}% reduction)")
        print(f"  Save time: {save_time:.1f}s")
    except Exception as e:
        print(f"✗ ERROR: Failed to save: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate by loading
    print(f"\n[4/4] Validating quantized checkpoint...")
    try:
        loaded_state = torch.load(output_path, map_location="cpu")
        print(f"✓ Checkpoint valid: {len(loaded_state)} tensors")
    except Exception as e:
        print(f"✗ ERROR: Checkpoint validation failed: {e}")
        return 1

    # Summary
    print(f"\n{'=' * 80}")
    print(f"✓ Quantization complete!")
    print(f"")
    print(f"  Input:  {input_path.name} ({input_size_gb:.2f} GB)")
    print(f"  Output: {output_path.name} ({output_size_gb:.2f} GB)")
    print(f"  Compression: {compression_ratio:.1f}×")
    print(f"")
    print(f"  Total time: {load_time + quant_time + save_time:.1f}s")
    print(f"")
    print(f"Next step:")
    print(f"  python run_quantized_inference.py \\")
    print(f"      --checkpoint {output_path.relative_to(SCRIPT_DIR)}")
    print(f"{'=' * 80}")

    return 0

if __name__ == "__main__":
    exit(main())
