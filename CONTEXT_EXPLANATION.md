# Why Pure Noise? Understanding Context in Diffusion Models

## The Problem

Current code produces **pure noise** because:

```python
context = torch.randn(1, 256, 4096)  # ❌ Random noise, no meaning
```

This is like feeding random gibberish to the model:
- Input: Random numbers
- Model: "I don't understand this, outputting random noise"
- Output: Pure noise

## What Context Should Be

Context is **semantic guidance** - it tells the model what to generate.

**Correct flow:**
```
Prompt: "A beautiful sunset over the ocean"
    ↓
Text Encoder (Qwen2.5-VL)
    ↓
Context: Dense vector representation of meaning
    ↓
DiT Model: "Generate video matching this concept"
    ↓
Output: Coherent video of sunset
```

## The Three Components

### 1. Text Encoder
Model that converts text → dense embeddings (2048 tokens, 4096 dims)

**Path:** `deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder/`

Check it exists:
```bash
ls -la deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder/
```

### 2. Tokenizer
Breaks text into tokens (word pieces)

```
"A beautiful sunset" → [101, 1037, 3376, 13891, 102]
```

### 3. Context Extraction
Take embeddings and pass to DiT model

```python
context = text_encoder(tokens).hidden_states[-1]  # Shape: (1, 256, 4096)
```

## How to Fix

### Option 1: Quick Test with Prompt

```bash
cd ~/JoyAI-Video-Edit
python3 test_quick_module.py "A beautiful landscape with mountains"
```

This will:
1. ✓ Load text encoder
2. ✓ Generate context from your prompt
3. ✓ Run diffusion with real guidance
4. ✓ Output should be coherent (not noise)

### Option 2: Full Inference with Prompt

```bash
cd ~/JoyAI-Video-Edit
python3 run_module.py input.mp4 output.mp4 "Your prompt here" 1 256 256 1
```

Arguments:
```
run_module.py <video> <output> <prompt> [frames] [height] [width] [steps]
```

### Option 3: Without Text Encoder (if not available)

If text encoder checkpoint doesn't exist, you'll get warnings but can still run:

```bash
python3 test_quick_module.py
```

This falls back to random context (output will be noise) but verifies the rest of the pipeline works.

## Debugging

### 1. Check if model is actually processing
```bash
python3 debug_inference.py
```

Look for:
```
✓ Output differs from input (model is processing)
```

### 2. Check if text encoder loads
```bash
python3 test_with_text.py
```

### 3. Check context quality
The context should have reasonable statistics:
```
Context mean: 0.1-0.5  (meaningful values)
Context std:  0.5-2.0  (some variance)
```

If context is random:
```
Context mean: 0.0      (looks random)
Context std:  1.0      (unit variance)
```

## Why This Matters

The DiT model was trained on **semantic guidance** (real text embeddings).

Without it:
- ❌ Model has no instruction
- ❌ Output is random noise
- ❌ All generated frames are garbage

With it:
- ✓ Model understands what to generate
- ✓ Output is coherent video
- ✓ Multiple prompts → different coherent outputs

## Summary

| Component | Status | Fix |
|-----------|--------|-----|
| DiT model | ✓ Loaded | Already done |
| VAE model | ✓ Loaded | Already done |
| Text encoder | ? Check | `ls deploy/deps/checkpoints/JoyAI-Video-Edit/text_encoder/` |
| Context generation | ✗ Random | Use prompt in `test_quick_module.py "your prompt"` |

**Bottom line:** Always provide a `--prompt` or pass text to generate context. Random context = random output.
