# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local smoke test for the JoyAI-Video-Edit streaming websocket endpoint.

Connects to /ws, sends a "start" message matching the exact payload
schema serve_joyomni_streaming.py's websocket_endpoint expects (see
its `elif msg_type == "start":` branch), and prints every JSON message
the server sends back (session_granted, session_reset, queue_position,
error, ...) so you can confirm the handshake without a browser.

Usage:
    python test_websocket_client.py --host 10.57.233.141 --port 8080
    python test_websocket_client.py --host 10.57.233.141 --ref-image assets/first_frame.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--prompt", default="a person walking through a city street")
    parser.add_argument(
        "--ref-image",
        type=Path,
        default=Path(__file__).parent / "assets" / "first_frame.png",
        help="Path to a local image to send as the session's reference frame.",
    )
    parser.add_argument(
        "--listen-seconds",
        type=float,
        default=8.0,
        help="How long to keep listening for server messages after sending 'start'.",
    )
    return parser.parse_args()


def encode_ref_image(path: Path) -> str | None:
    if not path.is_file():
        print(f"ref image not found at {path}, sending without one")
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


async def run(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}/ws"
    print(f"connecting to {uri}")
    async with websockets.connect(uri, max_size=None) as ws:
        start_payload = {
            "type": "start",
            "prompt": args.prompt,
            "ref_image": encode_ref_image(args.ref_image),
            "output_codec": "h264",
            "input_codec": "h264",
            "output_quality": 60,
            "gate_enabled": True,
        }
        print(f"sending: type={start_payload['type']!r} prompt={start_payload['prompt']!r} "
              f"has_ref_image={start_payload['ref_image'] is not None}")
        await ws.send(json.dumps(start_payload))

        try:
            async with asyncio.timeout(args.listen_seconds):
                async for message in ws:
                    if isinstance(message, bytes):
                        print(f"<< binary frame, {len(message)} bytes")
                        continue
                    payload = json.loads(message)
                    print(f"<< {payload}")
        except TimeoutError:
            print(f"done listening after {args.listen_seconds}s")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
