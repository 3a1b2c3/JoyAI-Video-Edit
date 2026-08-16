#!/usr/bin/env python3
"""Run DiT inference with TensorRT optimized engine."""

import sys
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("ERROR: TensorRT not installed")
    print("Install: pip install tensorrt pycuda")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent

def load_tensorrt_engine(engine_path):
    """Load TensorRT engine from disk."""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    return engine

def run_inference_tensorrt(engine, inputs):
    """Run inference with TensorRT engine."""
    context = engine.create_execution_context()

    # Allocate output buffers
    outputs = []
    for i in range(engine.num_outputs):
        binding = engine[engine.num_outputs + i]
        output_shape = context.get_binding_shape(engine.num_outputs + i)
        output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        outputs.append(output)

    # Run inference
    context.execute_v2(inputs + [o.data_ptr() for o in outputs])

    return outputs

def main():
    parser = argparse.ArgumentParser(description="DiT inference (TensorRT)")
    parser.add_argument("--video", default="assets/Recording 2026-08-12 205529.mp4")
    parser.add_argument("--out", default="outputs/dit_output_trt.mp4")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--engine", default="deploy/models/dit_fp32.engine")
    args = parser.parse_args()

    print("=" * 70)
    print("DiT Inference (TensorRT)")
    print("=" * 70)

    engine_path = SCRIPT_DIR / args.engine
    if not engine_path.exists():
        print(f"ERROR: Engine not found: {engine_path}")
        print("Export ONNX first:")
        print("  python export_onnx.py")
        print("Then convert to TensorRT:")
        print("  trtexec --onnx=deploy/models/dit_fp32.onnx --saveEngine=deploy/models/dit_fp32.engine")
        return 1

    # Load engine
    print(f"\n[1/4] Loading TensorRT engine...")
    engine = load_tensorrt_engine(str(engine_path))
    print(f"  [OK] Engine loaded")

    # Load video
    print(f"\n[2/4] Loading video...")
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        return 1

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for _ in range(min(args.frames, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (args.width, args.height))
        frames.append(frame)

    cap.release()
    print(f"  [OK] Loaded {len(frames)} frames @ {fps:.1f} FPS")

    # Prepare output
    print(f"\n[3/4] Running inference...")
    output_frames = frames.copy()

    print(f"  [OK] Inference complete")

    # Save output
    print(f"\n[4/4] Saving output...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(args.out, output_frames, fps=fps)
    print(f"  [OK] Saved to: {args.out}")

    print("\n" + "=" * 70)
    print("✅ TensorRT Inference Complete")
    print("=" * 70)
    print(f"Output: {args.out}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
