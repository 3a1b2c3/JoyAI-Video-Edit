#!/usr/bin/env python3
"""
Download JoyAI-Video-Edit models and dependencies.
Uses HF cache to avoid duplicates.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    uv_exe = Path("C:/Users/kschmid/.local/bin/uv.exe")
    subprocess.run([str(uv_exe), "pip", "install", "huggingface_hub", "-p", ".venv", "-q"], check=True)
    from huggingface_hub import hf_hub_download, snapshot_download

DEPLOY_DIR = Path("deploy")
CHECKPOINTS_DIR = DEPLOY_DIR / "deps" / "checkpoints"

# Use C:\Users\kschmid\.cache for HF models
HF_CACHE_DIR = Path("C:/Users/kschmid/.cache/huggingface")
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
HF_CACHE = HF_CACHE_DIR / "hub"

def check_hf_cache(repo_id, filename=None):
    """Check if model exists in HF cache, return path if found."""
    cache_dir = HF_CACHE
    if not cache_dir.exists():
        return None

    for model_dir in cache_dir.glob(f"models--{repo_id.replace('/', '--')}"):
        if filename:
            file_path = model_dir / "snapshots" / "*" / filename
            matches = list(model_dir.glob(f"snapshots/*/{filename}"))
            if matches:
                return matches[0]
        else:
            return model_dir

    return None

def link_or_copy(src, dst):
    """Link or copy file, avoiding duplication."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if src.is_file():
        print(f"    Using cached: {src}")
        try:
            # Try hard link first
            os.link(src, dst)
        except (OSError, NotImplementedError):
            # Fall back to copy
            shutil.copy2(src, dst)
    else:
        # Copy directory
        shutil.copytree(src, dst, dirs_exist_ok=True)

def main():
    print("=" * 70)
    print("JoyAI-Video-Edit Model Download")
    print("=" * 70)
    print(f"HF Cache: {HF_CACHE}")

    # Ensure directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JoyAI-Video-Edit weights (using 0811, latest)
    dit_dir = CHECKPOINTS_DIR / "JoyAI-Video-Edit" / "dit"
    dit_path = dit_dir / "joyai_video_edit_dit_0811.pth"

    if not dit_path.exists():
        print("\n1. JoyAI-Video-Edit DiT (0811)...")
        print("  Downloading from HF...")
        try:
            dit_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                "jdopensource/JoyAI-Video-Edit",
                "dit/joyai_video_edit_dit_0811.pth",
                cache_dir=str(HF_CACHE),
                local_dir=str(dit_dir.parent.parent)  # Points to JoyAI-Video-Edit
            )
            # Verify download
            if dit_path.exists():
                print(f"  ✓ Downloaded: {dit_path}")
            else:
                print(f"  ✗ File not found after download: {dit_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print(f"1. ✓ Found: {dit_path}")

    # 2. MiMo-VL-7B-RL-2508 (text encoder)
    mimo_path = CHECKPOINTS_DIR / "MiMo-VL-7B-RL-2508"
    if not mimo_path.exists():
        print("\n2. MiMo-VL-7B-RL-2508 (text encoder)...")

        cached = check_hf_cache("XiaomiMiMo/MiMo-VL-7B-RL-2508")
        if cached:
            print(f"  Found in cache: {cached}")
            link_or_copy(cached, mimo_path)
        else:
            print("  Downloading from HF...")
            try:
                snapshot_download(
                    "XiaomiMiMo/MiMo-VL-7B-RL-2508",
                    local_dir=mimo_path,
                    repo_type="model"
                )
                print("  ✓ Downloaded")
            except Exception as e:
                print(f"  ⚠ Failed: {e}")
    else:
        print(f"2. ✓ Found: {mimo_path}")

    # 3. YuNet face detector (optional)
    yunet_path = CHECKPOINTS_DIR / "face_detection_yunet_2023mar.onnx"
    if not yunet_path.exists():
        print("\n3. YuNet face detector (optional)...")

        try:
            print("  Downloading...")
            cached_file = hf_hub_download(
                "opencv/face_detection_yunet",
                "face_detection_yunet_2023mar.onnx",
                cache_dir=str(HF_CACHE)
            )
            link_or_copy(Path(cached_file), yunet_path)
            print("  ✓ Downloaded")
        except Exception as e:
            print(f"  ⚠ Failed: {e} (face detection disabled)")
    else:
        print(f"3. ✓ Found: {yunet_path}")

    # 4. YOLOv8n person detector (optional)
    yolo_path = CHECKPOINTS_DIR / "yolov8n.onnx"
    if not yolo_path.exists():
        print("\n4. YOLOv8n person detector (optional)...")

        try:
            print("  Installing ultralytics...")
            uv_exe = Path("C:/Users/kschmid/.local/bin/uv.exe")
            subprocess.run([str(uv_exe), "pip", "install", "ultralytics", "-p", ".venv", "-q"], check=True)

            print("  Downloading...")
            from ultralytics import YOLO
            model = YOLO("yolov8n.onnx")

            # Find the downloaded model
            import tempfile
            yolo_temp = Path(tempfile.gettempdir()) / "yolov8n.onnx"
            if yolo_temp.exists():
                link_or_copy(yolo_temp, yolo_path)
                print("  ✓ Downloaded")
            else:
                print("  ⚠ Model not found")
        except Exception as e:
            print(f"  ⚠ Failed: {e} (person detection disabled)")
    else:
        print(f"4. ✓ Found: {yolo_path}")

    # Verify
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    checks = [
        ((dit_dir / "joyai_video_edit_dit_0811.pth").exists(), f"JoyAI DiT 0811: {dit_dir / 'joyai_video_edit_dit_0811.pth'}"),
        ((CHECKPOINTS_DIR / "JoyAI-Video-Edit" / "vae").exists(), "JoyAI VAE"),
        (mimo_path.exists(), f"MiMo-VL: {mimo_path}"),
        (yunet_path.exists(), f"YuNet: {yunet_path}"),
        (yolo_path.exists(), f"YOLOv8n: {yolo_path}"),
    ]

    for exists, desc in checks:
        status = "✓" if exists else "✗"
        print(f"  {status} {desc}")

    print("\n" + "=" * 70)

    # Warn if DiT not available
    if not dit_path.exists():
        print("WARNING: DiT weights not found")
        print("=" * 70)
        print("\nCheck HF cache: python -c \"from pathlib import Path; print(list(Path('C:\\\\Users\\\\kschmid\\\\.cache\\\\huggingface\\\\hub').glob('*joyai*')))\"")
        print("Or download manually from: https://huggingface.co/jdopensource/JoyAI-Video-Edit")

    print("Download Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
