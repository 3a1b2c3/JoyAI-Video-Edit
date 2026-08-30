"""Standalone isolation test: does a CUDA graph capture attempt (mirroring
deploy/xvideo/serving/graph_runner.py's capture() pattern -- an eager
warmup call, then the SAME call again wrapped in torch.cuda.graph()) leave
the CUDA context/stream in a state where a SUBSEQUENT, otherwise-identical
eager call fails -- even though a fresh eager call worked fine before any
capture was attempted?

v1: plain bf16 nn.Linear only (no joyomni_ops/cutlass/FP8) -- ruled out as
a repro; capture+replay+post-capture-eager all succeed.

v2: added --loop (multi-step Python loop) and --kvcache (KV-cache-style
.copy_() writes into pre-allocated pool buffers) to get closer to the real
run_full()'s structure. Still plain nn.Linear -- did not reproduce either.

v3: adds --joyomni, which replaces the plain-Linear compute with the ACTUAL
joyomni_ops-backed ops dit.py calls unconditionally (regardless of the FP8
toggle): fused_layernorm_modulate, fused_qk_norm_rope_3d, rmsnorm_qk_bf16
(deploy/xvideo/models/dit/sgl_fused_ops.py). These were never exercised by
v1/v2, which only ever called cuBLAS via nn.Linear -- so v1/v2 "working"
was never evidence against a bug in these kernels specifically.

Usage:
    python test_cuda_graph_capture.py                     # default: joyomni+loop+kvcache (closest to real run_full())
    python test_cuda_graph_capture.py --no-joyomni --no-loop --no-kvcache  # v1 baseline (plain Linear, single call)
    python test_cuda_graph_capture.py --no-joyomni         # v2 (plain Linear, loop+kvcache)
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Self-contained: don't rely on the caller having PYTHONPATH set the way
# deploy/run_server.sh does (deploy/joyomni_ops first, then deploy, so the
# nested deploy/joyomni_ops/joyomni_ops/ package is what resolves instead of
# an empty outer namespace package -- see deploy/run_server.sh's PYTHONPATH
# fix).
sys.path.insert(0, os.path.join(SCRIPT_DIR, "deploy", "joyomni_ops"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "deploy"))

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_LAYERS = 4
NUM_STEPS = 4  # matches typical few-step denoising loop counts in this codebase

# Real dit.py dims (confirmed via [DEBUG fp8-fork] server log:
# img_attn_qkv.weight.shape=(12288, 4096) -> hidden_size=4096, qkv=12288).
HIDDEN_SIZE = 4096
HEADS_NUM = 32
HEAD_DIM = HIDDEN_SIZE // HEADS_NUM
SEQ_LEN = 4096
BATCH = 1
NORM_EPS = 1e-6


def build_model():
    lins = [torch.nn.Linear(3072, 9216, bias=True, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    return lins


def build_joyomni_inputs():
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    shift = torch.randn(BATCH, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    scale = torch.randn(BATCH, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    qkv_lin = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 3, bias=True, dtype=DTYPE, device=DEVICE)
    q_norm_weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k_norm_weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=DTYPE)
    # cos/sin: real code takes a (..., 1, L, head_dim) tuple, squeezes
    # leading size-1 dims, then -- since last dim == head_dim -- takes a
    # ::2 stride (paired rope). float32 like get_1d_rotary_pos_embed output.
    cos = torch.randn(1, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(1, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float32)
    return {
        "x": x, "shift": shift, "scale": scale, "qkv_lin": qkv_lin,
        "q_norm_weight": q_norm_weight, "k_norm_weight": k_norm_weight,
        "cos": cos, "sin": sin,
    }


def make_run_once_joyomni(inputs):
    from xvideo.models.dit import sgl_fused_ops as _sgl_fused

    x = inputs["x"]
    shift = inputs["shift"]
    scale = inputs["scale"]
    qkv_lin = inputs["qkv_lin"]
    q_norm_weight = inputs["q_norm_weight"]
    k_norm_weight = inputs["k_norm_weight"]
    cos = inputs["cos"]
    sin = inputs["sin"]

    def run_once():
        modulated = _sgl_fused.fused_layernorm_modulate(
            x, shift=shift, scale=scale, weight=None, bias=None, eps=NORM_EPS,
        )
        qkv = qkv_lin(modulated)
        qkv_v = qkv.view(qkv.shape[0], qkv.shape[1], 3, HEADS_NUM, HEAD_DIM)
        q, k, v = qkv_v[:, :, 0], qkv_v[:, :, 1], qkv_v[:, :, 2]
        k_for_cache = _sgl_fused.rmsnorm_qk_bf16(k, k_norm_weight, eps=NORM_EPS)
        q2, k2 = _sgl_fused.fused_qk_norm_rope_3d(
            q, k, q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
            freqs_cis=(cos, sin), eps=NORM_EPS,
        )
        return q2 + k2.to(q2.dtype) + v + k_for_cache.to(q2.dtype)

    return run_once


def make_run_fn(run_once, use_loop: bool, use_kvcache: bool, kv_shape):
    if use_kvcache:
        pool = torch.zeros(*kv_shape, device=DEVICE, dtype=DTYPE)
        stage = torch.zeros(*kv_shape, device=DEVICE, dtype=DTYPE)

    def do_kvcache_write(out):
        flat = out.reshape(-1)
        n = min(flat.numel(), stage.numel())
        stage.view(-1)[:n].copy_(flat[:n].to(DTYPE))
        pool.copy_(stage)

    def run_full():
        if not use_loop:
            out = run_once()
            if use_kvcache:
                do_kvcache_write(out)
            return out
        acc = None
        for _ in range(NUM_STEPS):
            out = run_once()
            acc = out if acc is None else acc + out
            if use_kvcache:
                do_kvcache_write(out)
        return acc

    return run_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True,
                         help="wrap in a multi-step Python loop (default: on)")
    parser.add_argument("--kvcache", action=argparse.BooleanOptionalAction, default=True,
                         help="add KV-cache-style .copy_() writes (default: on)")
    parser.add_argument("--joyomni", action=argparse.BooleanOptionalAction, default=True,
                         help="use real joyomni_ops kernels (fused_layernorm_modulate, "
                              "fused_qk_norm_rope_3d, rmsnorm_qk_bf16) instead of plain Linear "
                              "(default: on)")
    args = parser.parse_args()

    print(f"Config: loop={args.loop} kvcache={args.kvcache} joyomni={args.joyomni}")
    torch.cuda.set_device(0)

    if args.joyomni:
        inputs = build_joyomni_inputs()
        run_once = make_run_once_joyomni(inputs)
        kv_shape = (NUM_LAYERS, 4096, 128)
    else:
        lins = build_model()
        x = torch.randn(4096, 3072, device=DEVICE, dtype=DTYPE)

        def run_once():
            out = None
            for lin in lins:
                y = lin(x)
                out = y if out is None else out + y
            return out
        kv_shape = (NUM_LAYERS, 4096, 128)

    run_full = make_run_fn(run_once, args.loop, args.kvcache, kv_shape)

    print("[1] Baseline: fresh eager call, no capture attempted yet...")
    out = run_full()
    torch.cuda.synchronize()
    print(f"    OK, out shape={tuple(out.shape)}")

    print("[2] Attempting CUDA graph capture of the same call...")
    capture_ok = False
    try:
        mem_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=mem_pool, capture_error_mode="thread_local"):
            captured_out = run_full()
        torch.cuda.synchronize()
        print("    Capture succeeded. Replaying graph...")
        g.replay()
        torch.cuda.synchronize()
        print(f"    OK, replayed out shape={tuple(captured_out.shape)}")
        capture_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"    Capture FAILED: {e!r}")

    print("[3] Post-capture-attempt eager call (the critical test)...")
    try:
        out2 = run_full()
        torch.cuda.synchronize()
        print(f"    OK, out shape={tuple(out2.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED: {e!r}")
        print()
        if capture_ok:
            print("Capture succeeded but the eager call afterward still failed --")
            print("not a capture-failure-corrupts-state issue; something else changed.")
        else:
            print("CONFIRMED with this config: a failed graph capture attempt leaves")
            print("the CUDA context in a state where subsequent eager calls fail too.")
        raise

    print()
    print("Both capture attempt and post-capture eager call completed without error.")
    print("This config does NOT reproduce the server's crash.")


if __name__ == "__main__":
    main()
