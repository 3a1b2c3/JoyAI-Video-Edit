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

    # Try vision encoder
    print("Testing vision encoder...")
    with torch.no_grad():
        if hasattr(model, 'visual'):
            vision_model = model.visual
            print("  Using model.visual")
        else:
            vision_model = model.model.vision_model
            print("  Using model.model.vision_model")

        pixel_values = inputs['pixel_values'].to('cpu')
        print(f"  pixel_values shape: {pixel_values.shape}")

        image_features = vision_model(pixel_values)
        print(f"  vision_model output type: {type(image_features)}")

        if hasattr(image_features, 'last_hidden_state'):
            image_emb = image_features.last_hidden_state
        else:
            image_emb = image_features[0] if isinstance(image_features, tuple) else image_features

        print(f"  ✓ Image embedding shape: {image_emb.shape} dtype: {image_emb.dtype}")
        print()
        print(f"  Embedding stats:")
        print(f"    mean: {image_emb.mean():.6f}")
        print(f"    std:  {image_emb.std():.6f}")
        print(f"    min:  {image_emb.min():.6f}")
        print(f"    max:  {image_emb.max():.6f}")
        print()

    print("✅ Image encoding successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
