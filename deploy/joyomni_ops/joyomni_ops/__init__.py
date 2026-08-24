"""joyomni_ops — minimal self-contained CUDA ops for the JoyOmni V2V DiT pipeline.

Extracted from sgl-kernel; no sglang / sgl_kernel runtime dependency.
Exposes the ops the pipeline uses under a thin wrapper matching the original
sgl_kernel signatures, so call sites only change the import.
"""
import glob
import os
from typing import Optional

import torch

# The extension is a pure TORCH_LIBRARY (no pybind module init), so load the .so
# with torch.ops.load_library to trigger op registration.
_here = os.path.dirname(__file__)
_so = sorted(glob.glob(os.path.join(_here, "_C*.so")))
if not _so:
    raise ImportError(f"joyomni_ops C extension not found in {_here}; build it first (pip install .)")
if len(_so) > 1:
    # Picking [0] from an unordered glob silently risks loading a stale build left
    # behind by an earlier ABI tag / interrupted build -- fail loudly instead so this
    # never surfaces later as a confusing "cannot import name X (unknown location)".
    raise ImportError(
        f"Multiple _C*.so files found in {_here}: {_so} -- remove the stale one(s) "
        "and rebuild (bash build.sh) rather than silently picking one."
    )
try:
    torch.ops.load_library(_so[0])
except Exception as exc:
    # torch.ops.load_library failing here (e.g. dlopen "undefined symbol") leaves this
    # module partially initialized -- every name below is never defined, and callers
    # see a generic "cannot import name X from joyomni_ops (unknown location)" with no
    # hint of the real cause. Surface it directly instead.
    raise ImportError(f"joyomni_ops: torch.ops.load_library({_so[0]!r}) failed: {exc!r}") from exc

__all__ = [
    "fused_qk_norm_rope_3d_paired",
    "fused_norm_scale_shift",
    "rmsnorm",
    "sgl_per_token_quant_fp8",
    "fp8_scaled_mm",
    "has_fp8",
]

_ops = torch.ops.joyomni_ops


def fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, q_weight, k_weight, cos, sin):
    """In-place fused RMSNorm(q,k) + 3D RoPE. q/k: [B, seq_len*num_heads, head_dim] bf16."""
    _ops.fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, q_weight, k_weight, cos, sin)


def fused_norm_scale_shift(x, gamma, beta, scale, shift, norm_type, eps=1e-5):
    """Norm(x; gamma, beta) * (1 + scale) + shift. norm_type: 'layer' or 'rms'."""
    nt = 0 if norm_type == "layer" else 1 if norm_type == "rms" else -1
    return _ops.fused_norm_scale_shift(x, gamma, beta, scale, shift, nt, eps)


def rmsnorm(x, weight, eps=1e-6):
    """(x / RMS(x)) * weight. x: [M, N], weight: [N]."""
    return _ops.rmsnorm(x, weight, eps)


def has_fp8() -> bool:
    return hasattr(_ops, "fp8_scaled_mm")


def sgl_per_token_quant_fp8(input, output_q, output_s):
    _ops.sgl_per_token_quant_fp8(input, output_q, output_s)


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias: Optional[torch.Tensor] = None):
    return _ops.fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
