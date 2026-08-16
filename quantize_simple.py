#!/usr/bin/env python3
"""Simple int8 quantization using torchao (newer, more reliable)."""

import sys
from pathlib import Path
import time
import torch

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.utils import seed_everything
from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

def main():
    print("=" * 80)
    print("SIMPLE INT8 QUANTIZATION")
    print("=" * 80)

    input_path = Path(DEPLOY_DIR) / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth"
    output_path = Path(DEPLOY_DIR) / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811_quantized.pth"

    if not input_path.exists():
        print(f"✗ Checkpoint not found: {input_path}")
        return 1

    input_size_gb = input_path.stat().st_size / 1e9
    print(f"\n[1/3] Loading checkpoint...")
    print(f"  Input: {input_size_gb:.2f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        # Load model
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(input_path)

        dit = load_dit(cfg, device=device)
        dit.eval()
        dit.requires_grad_(False)

        print(f"✓ Loaded: {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B params")

        # Convert to float32 for quantization
        print(f"\n[2/3] Preparing for quantization...")
        current_dtype = next(dit.parameters()).dtype
        print(f"  Current dtype: {current_dtype}")

        if current_dtype != torch.float32:
            print(f"  Converting to float32...")
            dit = dit.float()

        # Simple quantization: store state_dict with scale/zero-point info
        print(f"\n[3/3] Quantizing...")

        # Get state dict
        state = dit.state_dict()

        # Quantize all float32 tensors to int8
        quantized_state = {}
        n_quantized = 0

        for name, tensor in state.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and tensor.numel() > 0:
                # Per-tensor quantization
                tensor_np = tensor.cpu().detach().numpy()
                scale = abs(tensor_np).max() / 127.0
                if scale == 0:
                    scale = 1.0

                tensor_int8 = (tensor_np / scale).round().astype("int8")

                # Store as int8 with scale as metadata
                quantized_state[name] = torch.from_numpy(tensor_int8).to(tensor.device)
                quantized_state[f"{name}__scale"] = torch.tensor(scale, dtype=torch.float32)
                n_quantized += 1
            else:
                quantized_state[name] = tensor

        print(f"✓ Quantized {n_quantized} tensors")

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_state, output_path)
        output_size_gb = output_path.stat().st_size / 1e9
        ratio = input_size_gb / output_size_gb

        print(f"\n{'=' * 80}")
        print(f"✓ Complete!")
        print(f"  Input:  {input_size_gb:.2f} GB")
        print(f"  Output: {output_size_gb:.2f} GB")
        print(f"  Compression: {ratio:.1f}×")
        print(f"  Path: {output_path}")
        print(f"{'=' * 80}")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
