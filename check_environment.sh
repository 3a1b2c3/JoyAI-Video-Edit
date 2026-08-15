#!/bin/bash
# Check GPU and environment setup

echo "========================================================================"
echo "Environment Check for DiT Inference"
echo "========================================================================"
echo ""

# Check Python
echo "[1/5] Python:"
if command -v python &> /dev/null; then
    python --version
    python -c "import sys; print('  Location:', sys.executable)"
else
    echo "  ❌ Python not found"
fi
echo ""

# Check PyTorch
echo "[2/5] PyTorch:"
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print('  Version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
else
    echo "  ❌ PyTorch not installed"
fi
echo ""

# Check GPU
echo "[3/5] GPU Memory:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,nounits,noheader
else
    echo "  ❌ nvidia-smi not found"
fi
echo ""

# Check checkpoints
echo "[4/5] Checkpoints:"
if [ -f "deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth" ]; then
    SIZE=$(du -h "deploy/deps/checkpoints/JoyAI-Video-Edit/dit/dit/joyai_video_edit_dit_0804.pth" | cut -f1)
    echo "  ✅ Float32 DiT: $SIZE"
else
    echo "  ❌ Float32 DiT checkpoint not found"
fi

if [ -d "deploy/deps/checkpoints/JoyAI-Video-Edit/vae" ]; then
    echo "  ✅ VAE checkpoint found"
else
    echo "  ❌ VAE checkpoint not found"
fi
echo ""

# Check inference script
echo "[5/5] Scripts:"
if [ -f "run_inference.py" ]; then
    echo "  ✅ run_inference.py"
else
    echo "  ❌ run_inference.py not found"
fi
echo ""

echo "========================================================================"
echo "To run inference:"
echo "  bash run_inference.sh"
echo ""
echo "For custom parameters:"
echo "  bash run_inference_custom.sh input.mp4 output.mp4 2 256 256 4 42"
echo ""
echo "For batch processing:"
echo "  bash batch_inference.sh ./videos outputs/batch 2 1"
echo "========================================================================"
