"""Standalone isolation test: does a CUDA graph capture attempt (mirroring
deploy/xvideo/serving/graph_runner.py's capture() pattern -- an eager
warmup call, then the SAME call again wrapped in torch.cuda.graph()) leave
the CUDA context/stream in a state where a SUBSEQUENT, otherwise-identical
eager call fails -- even though a fresh eager call worked fine before any
capture was attempted?

No joyomni_ops/cutlass/FP8 involved -- plain bf16 nn.Linear only, to
isolate whether this is a CUDA-graph-capture issue independent of FP8.

v2: adds the two things v1 (a single Linear call) didn't have, which the
real run_full() in graph_runner.py does: a Python-level multi-step loop,
and KV-cache-style .copy_() writes into pre-allocated pool buffers. Each
is toggleable via CLI flag so we can bisect which one (if either) is what
actually breaks capture on this GPU.

Usage:
    python test_cuda_graph_capture.py                    # v1 behavior (baseline)
    python test_cuda_graph_capture.py --loop              # + multi-step loop
    python test_cuda_graph_capture.py --loop --kvcache     # + KV-cache-style copies
"""
import argparse

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_LAYERS = 4
NUM_STEPS = 4  # matches typical few-step denoising loop counts in this codebase


def build_model():
    lins = [torch.nn.Linear(3072, 9216, bias=True, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    return lins


def make_run_fn(lins, x, use_loop: bool, use_kvcache: bool):
    if use_kvcache:
        # Mirrors graph_runner.py's pool_k/pool_v (static, pre-allocated,
        # written into via .copy_() from a per-layer "stage" tensor) and
        # commit_cos/commit_sin (a small buffer swapped per chunk).
        pool = torch.zeros(NUM_LAYERS, 4096, 128, device=DEVICE, dtype=DTYPE)
        stage = torch.zeros(NUM_LAYERS, 4096, 128, device=DEVICE, dtype=DTYPE)

    def run_once():
        # Each layer applied independently to the same input (not chained --
        # real per-block Linears project 3072->9216 then back down to 3072
        # via attention/proj before the next block; chaining identical
        # Linear(3072,9216) layers here would be a shape mismatch, and isn't
        # the thing under test anyway).
        out = None
        for lin in lins:
            y = lin(x)
            out = y if out is None else out + y
        return out

    def run_full():
        if not use_loop:
            out = run_once()
            if use_kvcache:
                for li in range(NUM_LAYERS):
                    stage[li].copy_(out[:4096, :128])
                    pool[li].copy_(stage[li])
            return out
        acc = None
        for step in range(NUM_STEPS):
            out = run_once()
            acc = out if acc is None else acc + out
            if use_kvcache:
                for li in range(NUM_LAYERS):
                    stage[li].copy_(out[:4096, :128])
                    pool[li].copy_(stage[li])
        return acc

    return run_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="wrap in a multi-step Python loop")
    parser.add_argument("--kvcache", action="store_true", help="add KV-cache-style .copy_() writes")
    args = parser.parse_args()

    print(f"Config: loop={args.loop} kvcache={args.kvcache}")
    torch.cuda.set_device(0)
    lins = build_model()
    x = torch.randn(4096, 3072, device=DEVICE, dtype=DTYPE)
    run_full = make_run_fn(lins, x, args.loop, args.kvcache)

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
    print("This config does NOT reproduce the server's crash. Try adding --loop and/or")
    print("--kvcache if not already set, to get closer to the real run_full().")


if __name__ == "__main__":
    main()
