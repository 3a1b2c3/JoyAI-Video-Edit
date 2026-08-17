# Shell Scripts for DiT Inference

Quick bash scripts to run inference without typing long Python commands.

## Files

### 1. `check_environment.sh`
Check GPU, PyTorch, and checkpoint setup before running inference.

```bash
bash check_environment.sh
```

**Output**:
- Python version
- PyTorch & CUDA status
- GPU memory available
- Checkpoint locations
- Setup instructions

---

### 2. `run_inference.sh`
Basic inference with sensible defaults.

```bash
bash run_inference.sh
```

**Arguments** (all optional):
```bash
bash run_inference.sh [input_video] [output_video] [frames] [steps]
```

**Examples**:
```bash
# Default: 2 frames, 1 step
bash run_inference.sh

# Custom input/output
bash run_inference.sh my_video.mp4 my_output.mp4

# With frame and step count
bash run_inference.sh my_video.mp4 output.mp4 4 8
```

---

### 3. `run_inference_custom.sh`
Full control over all parameters with progress display.

```bash
bash run_inference_custom.sh [video] [output] [frames] [height] [width] [steps] [seed]
```

**Parameters**:
- `video` — Input video path
- `output` — Output video path
- `frames` — Number of frames to process
- `height` — Frame height (pixels)
- `width` — Frame width (pixels)
- `steps` — Denoising steps (1-16)
- `seed` — Random seed for reproducibility

**Example**:
```bash
# 720p, 4 frames, 8 denoising steps
bash run_inference_custom.sh input.mp4 output.mp4 4 720 1280 8 42
```

---

### 4. `batch_inference.sh`
Process multiple videos in a directory.

```bash
bash batch_inference.sh [input_dir] [output_dir] [frames] [steps] [pattern]
```

**Parameters**:
- `input_dir` — Directory containing videos (default: current directory)
- `output_dir` — Where to save outputs (default: `outputs/batch`)
- `frames` — Frames per video (default: 2)
- `steps` — Denoising steps (default: 1)
- `pattern` — File pattern to match (default: `*.mp4`)

**Examples**:
```bash
# Process all .mp4 in ./videos
bash batch_inference.sh ./videos outputs/batch 2 4

# Custom file pattern (.mov files)
bash batch_inference.sh ./videos outputs "*.mov" 2 4 "*.mov"
```

**Output**:
- Processes each video in order
- Saves to `outputs/batch/` with `_dit` suffix
- Progress display for each file

---

## Usage Workflow

### 1. Check Setup
```bash
bash check_environment.sh
```

### 2. Test Single Video
```bash
bash run_inference.sh
```

### 3. Run with Custom Settings
```bash
bash run_inference_custom.sh my_video.mp4 my_output.mp4 4 512 512 8 42
```

### 4. Batch Process
```bash
bash batch_inference.sh ./input_videos outputs/results 4 4
```

---

## Environment

Scripts automatically:
- ✅ Find and activate `.venv` if present
- ✅ Set working directory correctly
- ✅ Show progress and timing
- ✅ Display output file size and location
- ✅ Handle errors gracefully

## Requirements

- Bash (Linux, macOS, or WSL on Windows)
- Python 3.10+ with PyTorch
- GPU with CUDA support
- ~30 GB VRAM for RTX 5090, 48 GB for A40

## Tips

**High Quality (Slower)**:
```bash
bash run_inference_custom.sh input.mp4 output.mp4 8 1080 1920 16 42
```

**Fast (Lower Quality)**:
```bash
bash run_inference.sh input.mp4 output.mp4 2 1
```

**Batch with High Quality**:
```bash
bash batch_inference.sh ./videos outputs/hq 4 8
```

---

## Troubleshooting

**"command not found: bash"**
- You're on Windows without WSL/Git Bash
- Use Python directly: `python run_inference.py`

**"Permission denied"**
- Make scripts executable: `chmod +x *.sh`

**"venv not found"**
- Create venv: `python -m venv .venv`
- Or install dependencies: `pip install -r requirements.txt`

**"GPU OOM"**
- Reduce `frames` or resolution
- Check `nvidia-smi` for other processes using GPU

---

## Summary

| Script | Purpose | Use Case |
|--------|---------|----------|
| `check_environment.sh` | Verify setup | Before first run |
| `run_inference.sh` | Quick test | Single video, defaults |
| `run_inference_custom.sh` | Full control | Fine-tuning parameters |
| `batch_inference.sh` | Process many | Directory of videos |
