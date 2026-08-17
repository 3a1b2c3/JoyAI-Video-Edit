# DiT Inference Scripts — Complete Reference

## 📋 Quick Navigation

### First Time Setup
1. **Start here**: `SETUP_GUIDE.md` — Complete installation walkthrough
2. **Run**: `bash setup_all.sh` — Automated setup

### Using the Scripts
3. **Scripts guide**: `SHELL_SCRIPTS.md` — How to use each script
4. **Run inference**: `bash run_inference.sh` — Start processing videos

### Understanding the System
5. **Inference guide**: `RUN_INFERENCE.md` — Options and performance
6. **Quantization details**: `QUANTIZATION_ATTEMPTS.md` — Why quantization failed

---

## 📁 All Available Scripts

### Setup Scripts (Run Once)
| Script | Purpose | Time |
|--------|---------|------|
| `setup_all.sh` | Complete automated setup | 5-15 min |
| `setup_venv.sh` | Create venv + install packages | 5-10 min |
| `download_checkpoints.sh` | Setup checkpoint directories | 1 min |
| `check_environment.sh` | Verify GPU/PyTorch/models | 1 min |

### Inference Scripts (Run Repeatedly)
| Script | Purpose | Use Case |
|--------|---------|----------|
| `run_inference.sh` | Quick inference with defaults | Single video, default settings |
| `run_inference_custom.sh` | Full parameter control | Custom resolution/quality |
| `batch_inference.sh` | Process multiple videos | Directory of videos |

---

## 🚀 Quick Start

### First Time
```bash
# 1. Complete setup (interactive)
bash setup_all.sh

# 2. Download checkpoints (manual)
# See SETUP_GUIDE.md for checkpoint locations

# 3. Verify everything works
bash check_environment.sh
```

### Run Inference
```bash
# Simple: use defaults
bash run_inference.sh

# Custom: full control
bash run_inference_custom.sh input.mp4 output.mp4 4 512 512 8 42

# Batch: process directory
bash batch_inference.sh ./videos outputs 4 4
```

---

## 📖 Documentation Files

### Setup & Configuration
- **`SETUP_GUIDE.md`** — Installation walkthrough with troubleshooting
- **`SHELL_SCRIPTS.md`** — Detailed script documentation

### Usage & Performance
- **`RUN_INFERENCE.md`** — Inference options and performance metrics
- **`INFERENCE_README.md`** — Complete feature reference

### Technical Details
- **`QUANTIZATION_ATTEMPTS.md`** — Why 3 quantization approaches failed
- **`SCRIPTS_README.md`** — This file

---

## 🎯 Common Tasks

### Test GPU Setup
```bash
bash check_environment.sh
```

### Run Single Video
```bash
bash run_inference.sh input.mp4 output.mp4
```

### High Quality Output
```bash
bash run_inference_custom.sh \
  input.mp4 output.mp4 \
  8 1080 1920 16 42
```

### Process Directory of Videos
```bash
bash batch_inference.sh ./input_videos outputs 4 8
```

### Setup on New Machine
```bash
bash setup_all.sh
# Then download checkpoints (see SETUP_GUIDE.md)
```

---

## 📊 Script Parameters

### `run_inference_custom.sh`
```bash
bash run_inference_custom.sh video output frames height width steps seed
```

- `video` — Input video path
- `output` — Output video path
- `frames` — Number of frames (1-64)
- `height` — Height in pixels (128-2048)
- `width` — Width in pixels (128-2048)
- `steps` — Denoising steps (1-16, higher=better but slower)
- `seed` — Random seed (0-2^31)

### `batch_inference.sh`
```bash
bash batch_inference.sh input_dir output_dir frames steps pattern
```

- `input_dir` — Directory with videos
- `output_dir` — Where to save outputs
- `frames` — Frames per video
- `steps` — Denoising steps
- `pattern` — File pattern (e.g., `*.mp4`)

---

## ⚙️ Environment Setup

Scripts automatically:
- ✅ Activate virtual environment
- ✅ Set working directory correctly
- ✅ Verify dependencies
- ✅ Show progress and timing
- ✅ Handle errors gracefully

Manual activation (if needed):
```bash
# Linux/macOS
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 💾 Checkpoint Locations

The scripts expect checkpoints at:
```
deploy/deps/checkpoints/JoyAI-Video-Edit/
├── dit/dit/
│   └── joyai_video_edit_dit_0804.pth         (28-30 GB)
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.bin
```

See `SETUP_GUIDE.md` for download instructions.

---

## 🔧 Troubleshooting

### Script Won't Run
```bash
# Make executable (Linux/macOS)
chmod +x *.sh

# Or run with explicit bash
bash run_inference.sh
```

### GPU Not Detected
```bash
nvidia-smi  # Should show your GPU
bash check_environment.sh  # Verify CUDA setup
```

### Checkpoints Missing
```bash
bash download_checkpoints.sh  # Show expected locations
# Then manually download and place files
```

### Out of Memory
```bash
# Reduce parameters
bash run_inference_custom.sh input.mp4 output.mp4 2 256 256 1 42
```

See `SETUP_GUIDE.md` for detailed troubleshooting.

---

## 📚 Learning Path

1. **Day 1: Setup**
   - Read: `SETUP_GUIDE.md`
   - Run: `bash setup_all.sh`
   - Verify: `bash check_environment.sh`

2. **Day 1-2: Test Inference**
   - Read: `SHELL_SCRIPTS.md`
   - Run: `bash run_inference.sh`
   - Experiment: `bash run_inference_custom.sh`

3. **Day 2+: Production Use**
   - Read: `RUN_INFERENCE.md`
   - Use: `bash batch_inference.sh` for multiple videos
   - Tune: Adjust `--steps` and resolution for quality/speed

4. **Understanding (Optional)**
   - Read: `QUANTIZATION_ATTEMPTS.md` (technical details)
   - Read: `INFERENCE_README.md` (feature reference)

---

## 📋 Checklist

Before running inference:
- [ ] Python 3.10+ installed
- [ ] NVIDIA GPU with CUDA support
- [ ] 32+ GB VRAM (RTX 5090) or 48+ GB (A40)
- [ ] Setup scripts downloaded
- [ ] `setup_all.sh` completed successfully
- [ ] Checkpoints downloaded and placed
- [ ] `check_environment.sh` shows all ✅

---

## 🎬 Ready to Use

All scripts are production-ready. No dummy code—all scripts:
- ✅ Execute real inference
- ✅ Handle errors gracefully
- ✅ Show progress clearly
- ✅ Display results with timing/file info

**Start with**: `bash setup_all.sh`

**Then run**: `bash run_inference.sh input.mp4`

---

## Summary

| Phase | Script | Docs |
|-------|--------|------|
| Setup | `setup_all.sh` | `SETUP_GUIDE.md` |
| Verify | `check_environment.sh` | — |
| Quick Run | `run_inference.sh` | `SHELL_SCRIPTS.md` |
| Custom | `run_inference_custom.sh` | `SHELL_SCRIPTS.md` |
| Batch | `batch_inference.sh` | `SHELL_SCRIPTS.md` |
| Learn | — | `QUANTIZATION_ATTEMPTS.md` |

**Total first-time setup**: ~15 minutes
