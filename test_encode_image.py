#!/usr/bin/env python3
"""Test Qwen2.5-VL image encoding only"""

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print("Testing Qwen2.5-VL image encoding...")
print()

# Load model
print("[1/3] Loading Qwen2.5-VL...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=False)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    local_files_only=False,
    torch_dtype=torch.float16,
    device_map="cpu"
)
model.eval()
print("✓ Model loaded")
print()

# Load image
print("[2/3] Loading image...")
img_path = "assets/image.png"
if not Path(img_path).exists():
    print(f"ERROR: {img_path} not found")
    exit(1)

style_img = Image.open(img_path).convert("RGB")
print(f"✓ Image loaded: {style_img.size}")
print()

# Test encoding
print("[3/3] Testing image encoding...")
try:
    # Process image (text=empty)
    inputs = processor(images=[style_img], return_tensors="pt")
    print(f"✓ Processor output:")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape} {v.dtype}")
    print()

    # Test full model forward with image tokens
    print("Testing full model with image tokens...")
    prompt_text = "Describe the visual style of this image.\n<|vision_start|><|image_pad|><|vision_end|>"

    inputs = processor(text=prompt_text, images=[style_img], return_tensors="pt")
    print(f"  Processor output keys: {list(inputs.keys())}")

    with torch.no_grad():
        outputs = model(**{k: v.to('cpu') for k, v in inputs.items()}, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

        print(f"  ✓ Embeddings shape: {embeddings.shape} dtype: {embeddings.dtype}")
        print()
        print(f"  Embedding stats:")
        print(f"    mean: {embeddings.mean():.6f}")
        print(f"    std:  {embeddings.std():.6f}")
        print(f"    min:  {embeddings.min():.6f}")
        print(f"    max:  {embeddings.max():.6f}")
        print()

    print("✅ Image encoding successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
