import os
import subprocess
import sys
from pathlib import Path


repository_root = Path(__file__).resolve().parents[1]

checkpoint_root = Path(
    os.getenv(
        "JOYOMNI_CKPT_ROOT",
        "/runpod-volume/joyai/checkpoints",
    )
)

required_items = [
    checkpoint_root
    / "JoyAI-Video-Edit"
    / "dit"
    / "joyai_video_edit_dit_0811.pth",

    checkpoint_root
    / "JoyAI-Video-Edit"
    / "vae"
    / "config.json",

    checkpoint_root
    / "JoyAI-Video-Edit"
    / "vae"
    / "diffusion_pytorch_model.safetensors",

    checkpoint_root
    / "MiMo-VL-7B-RL-2508",
]

missing_items = [
    str(item)
    for item in required_items
    if not item.exists()
]

if missing_items:
    print("Required model files are missing:", flush=True)

    for item in missing_items:
        print(f" - {item}", flush=True)

    print(
        "\nRun this command once on the attached RunPod volume:",
        flush=True,
    )
    print(
        "python3 /opt/joyai/runpod/download_models.py",
        flush=True,
    )

    raise SystemExit(1)

environment = os.environ.copy()

# RunPod supplies PORT for public traffic.
model_port = environment.get(
    "PORT",
    environment.get("JOYOMNI_PORT", "8080"),
)

environment["JOYOMNI_PORT"] = model_port

print(
    f"Starting JoyAI on port {model_port}...",
    flush=True,
)

health_process = subprocess.Popen(
    [
        sys.executable,
        str(repository_root / "runpod" / "health_server.py"),
    ],
    cwd=repository_root,
    env=environment,
)

exit_code = 1

try:
    model_process = subprocess.Popen(
        [
            "bash",
            str(repository_root / "deploy" / "run_server.sh"),
        ],
        cwd=repository_root,
        env=environment,
    )

    exit_code = model_process.wait()

finally:
    health_process.terminate()

    try:
        health_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        health_process.kill()

raise SystemExit(exit_code)