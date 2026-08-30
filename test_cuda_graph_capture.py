"""Standalone isolation test: does a CUDA graph capture attempt (which
mirrors deploy/xvideo/serving/graph_runner.py's capture() pattern -- an
eager warmup call, then the SAME call again wrapped in torch.cuda.graph())
leave the CUDA context/stream in a state where a SUBSEQUENT, otherwise-
identical eager call fails -- even though a fresh eager call (skipped the
capture attempt entirely) is already known to work fine on this GPU?

No joyomni_ops/cutlass/FP8 involved at all -- plain bf16 nn.Linear only, to
isolate whether this is a CUDA-graph-capture issue independent of FP8.

Usage:
    python test_cuda_graph_capture.py
"""
import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16


def run_linear(lin: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return lin(x)


def main() -> None:
    torch.cuda.set_device(0)
    lin = torch.nn.Linear(3072, 9216, bias=True, dtype=DTYPE, device=DEVICE)
    x = torch.randn(4096, 3072, device=DEVICE, dtype=DTYPE)

    print("[1] Baseline: fresh eager call, no capture attempted yet...")
    out = run_linear(lin, x)
    torch.cuda.synchronize()
    print(f"    OK, out shape={tuple(out.shape)}")

    print("[2] Attempting CUDA graph capture of the same call...")
    capture_ok = False
    try:
        mem_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=mem_pool, capture_error_mode="thread_local"):
            captured_out = run_linear(lin, x)
        torch.cuda.synchronize()
        print(f"    Capture succeeded. Replaying graph...")
        g.replay()
        torch.cuda.synchronize()
        print(f"    OK, replayed out shape={tuple(captured_out.shape)}")
        capture_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"    Capture FAILED: {e!r}")

    print("[3] Post-capture-attempt eager call (the critical test)...")
    try:
        out2 = run_linear(lin, x)
        torch.cuda.synchronize()
        print(f"    OK, out shape={tuple(out2.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED: {e!r}")
        print()
        if capture_ok:
            print("Capture succeeded but the eager call afterward still failed --")
            print("not a capture-failure-corrupts-state issue; something else changed.")
        else:
            print("CONFIRMED: a failed graph capture attempt leaves the CUDA context")
            print("in a state where subsequent eager calls fail too, even though a")
            print("fresh eager call (step 1) worked fine before any capture was tried.")
        raise

    print()
    print("Both capture attempt and post-capture eager call completed without error.")
    print("This minimal case does NOT reproduce the server's crash -- the real")
    print("failure needs something the server does that this simplified script")
    print("doesn't (dynamic control flow, KV-cache indexing, multi-step loop, etc).")


if __name__ == "__main__":
    main()
