#!/usr/bin/env python3
"""Simple per-tensor int8 quantization for large models (16B+ params)"""

import torch
import sys
from pathlib import Path

def quantize_tensor(tensor):
    """Quantize single tensor to int8 with scale factor"""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return tensor

    # Convert to float32 for quantization
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    # Get scale factor (max value / 127)
    max_val = tensor.abs().max().item()
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127.0

    # Quantize to int8
    quantized = (tensor / scale).round().to(torch.int8)

    # Store scale as metadata in a dict for dequantization
    return {"data": quantized, "scale": scale}

def dequantize_tensor(quant_dict):
    """Dequantize int8 tensor back to float32"""
    if not isinstance(quant_dict, dict) or "data" not in quant_dict:
        return quant_dict

    quantized = quant_dict["data"]
    scale = quant_dict["scale"]
    return (quantized.float() * scale)

def quantize_checkpoint(checkpoint_path, output_path):
    """Load checkpoint, quantize, and save"""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    total_size_before = sum(
        p.numel() * p.element_size() for p in state_dict.values()
        if isinstance(p, torch.Tensor)
    ) / 1e9

    print(f"Original size: {total_size_before:.2f}GB")
    print("Quantizing tensors...")

    quantized_dict = {}
    for i, (k, v) in enumerate(state_dict.items()):
        if isinstance(v, torch.Tensor):
            quantized_dict[k] = quantize_tensor(v)
        else:
            quantized_dict[k] = v

        if (i + 1) % 100 == 0:
            print(f"  Quantized {i + 1} tensors...")

    total_size_after = sum(
        q["data"].numel() * q["data"].element_size() if isinstance(q, dict) else 0
        for q in quantized_dict.values()
    ) / 1e9

    print(f"Quantized size: {total_size_after:.2f}GB")
    print(f"Compression: {total_size_before / total_size_after:.2f}x")

    print(f"Saving to: {output_path}")
    torch.save(quantized_dict, output_path)
    print("✅ Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quantize_simple.py <input_ckpt> <output_ckpt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    quantize_checkpoint(input_path, output_path)
