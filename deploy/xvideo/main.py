"""FastAPI app wrapper for JoyAI-Video-Edit server."""

import argparse
import sys
from pathlib import Path

# Get repo root (parent of deploy/)
REPO_ROOT = Path(__file__).resolve().parents[2]

from xvideo.serving.serve_joyomni_streaming import create_app

# Create args for create_app
args = argparse.Namespace(
    dit_ckpt=str(REPO_ROOT / "deploy" / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "dit" / "joyai_video_edit_dit_0804.pth"),
    vae_ckpt=str(REPO_ROOT / "deploy" / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "vae"),
    text_encoder_ckpt=str(REPO_ROOT / "deploy" / "deps" / "checkpoints" / "MiMo-VL-7B-RL-2508"),
    face_detector_onnx=str(REPO_ROOT / "deploy" / "deps" / "checkpoints" / "face_detection_yunet_2023mar.onnx"),
    person_detector_onnx=str(REPO_ROOT / "deploy" / "deps" / "checkpoints" / "yolov8n.onnx"),
    enable_persona_gate=True,
    enable_face_gate=True,
    pe_api_key=None,
    pe_base_url=None,
    pe_model=None,
    device="cuda:0",
    dtype="bfloat16",
    streaming_mode="streaming",
    num_inference_steps=8,
    preload=False,
    profile_timings=False,
    port=8080,
)

app = create_app(args)
