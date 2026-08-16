#!/usr/bin/env python3
"""Export DiT to ONNX for TensorRT."""

import sys
from pathlib import Path
import torch
import torch.onnx

SCRIPT_DIR = Path(__file__).parent
DEPLOY_DIR = SCRIPT_DIR / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

from xvideo.models.models import load_dit
from xvideo.config import ExpConfig

def export_dit_to_onnx():
    print("=" * 70)
    print("Export DiT to ONNX for TensorRT")
    print("=" * 70)

    device = torch.device("cuda")

    # Load config and model
    print("\n[1/3] Loading DiT model...")
    try:
        cfg = ExpConfig()
        cfg.training_mode = False
        cfg.dit_ckpt = str(DEPLOY_DIR / "deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0811.pth")

        dit = load_dit(cfg, device=device)
        dit.eval()
        dit.requires_grad_(False)

        # Force all parameters and buffers to float32
        for param in dit.parameters():
            param.data = param.data.to(dtype=torch.float32)
        for buf in dit.buffers():
            if buf.dtype in [torch.float16, torch.bfloat16]:
                buf.data = buf.data.to(dtype=torch.float32)

        print(f"  [OK] Model loaded (converted to float32)")

        # Get actual input shapes from config
        batch_size = 1
        num_frames = getattr(cfg, 'num_frames', 16)
        in_channels = getattr(cfg, 'in_channels', 4)
        height = getattr(cfg, 'height', 256)
        width = getattr(cfg, 'width', 256)

        print(f"  Input shape: ({batch_size}, {num_frames}, {in_channels}, {height}, {width})")
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        print("  Fallback: Use PyTorch float32 inference instead")
        return False

    # Create dummy inputs matching actual inference shapes
    print("\n[2/3] Creating dummy inputs...")
    dummy_x = torch.randn(batch_size, in_channels, num_frames, height, width, device=device, dtype=torch.float32)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device)
    dummy_encoder_hidden_states = torch.randn(batch_size, 256, 4096, device=device, dtype=torch.float32)

    print(f"  [OK] Dummy inputs created")
    print(f"    x: {dummy_x.shape}")
    print(f"    t: {dummy_t.shape}")
    print(f"    encoder_hidden_states: {dummy_encoder_hidden_states.shape}")

    # Export to ONNX
    print("\n[3/3] Exporting to ONNX...")
    output_path = str(SCRIPT_DIR / "deploy/models/dit_fp32.onnx")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            dit,
            (dummy_x, dummy_t, dummy_encoder_hidden_states),
            output_path,
            input_names=["x", "t", "encoder_hidden_states"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "batch_size", 2: "num_frames"},
                "t": {0: "batch_size"},
                "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"  [OK] Exported to: {output_path}")
    except Exception as e:
        print(f"  ERROR: Export failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nFallback: Use PyTorch float32 inference instead")
        return False

    print("\n" + "=" * 70)
    print("ONNX Export Complete")
    print("=" * 70)
    print(f"Model: {output_path}")
    print("\nNext: Convert to TensorRT")
    print("  trtexec --onnx=deploy/models/dit_fp32.onnx --saveEngine=deploy/models/dit_fp32.engine")

    return True

if __name__ == "__main__":
    success = export_dit_to_onnx()
    sys.exit(0 if success else 1)
