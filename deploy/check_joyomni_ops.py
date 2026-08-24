#!/usr/bin/env python3
"""Prerequisite check: verify joyomni_ops is a real, importable install for this interpreter.

Run this before infer_standalone.py / the server. Catches the failure mode where
`import joyomni_ops` silently resolves to a namespace package assembled from
source directories on sys.path (e.g. the project's top-level joyomni_ops/ folder)
instead of the actual compiled extension -- which happens when the .so was built
for a different Python version/ABI than the one currently running, or was never
pip-installed into this environment.
"""
import sys
import sysconfig


def main() -> int:
    print(f"Interpreter: {sys.executable}")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"ABI tag:     {sysconfig.get_config_var('EXT_SUFFIX')}")
    print()

    try:
        import joyomni_ops
    except ImportError as exc:
        print(f"FAIL: cannot import joyomni_ops: {exc!r}")
        print()
        print("Build/install it for THIS interpreter:")
        print(f"  cd joyomni_ops && PYTHON={sys.executable} bash build.sh")
        print(f"  # or: {sys.executable} -m pip install -e joyomni_ops --no-build-isolation --force-reinstall")
        return 1

    module_file = getattr(joyomni_ops, "__file__", None)
    if not module_file:
        print("FAIL: joyomni_ops resolved as a namespace package (no __file__) -- not a real install.")
        print(f"  sys.path entries matched: {list(getattr(joyomni_ops, '__path__', []))}")
        print()
        print("This means a bare joyomni_ops/ directory earlier on sys.path is shadowing the real")
        print("package (which was never pip-installed for this interpreter, or was built for a")
        print("different Python version). Rebuild for this interpreter:")
        print(f"  cd joyomni_ops && PYTHON={sys.executable} bash build.sh")
        return 1

    print(f"Package file: {module_file}")

    try:
        from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
    except ImportError as exc:
        print(f"FAIL: joyomni_ops is a real package but is missing expected ops: {exc!r}")
        print("  The installed .so is likely stale or built without the required kernels.")
        print(f"  Rebuild: cd joyomni_ops && PYTHON={sys.executable} bash build.sh")
        return 1

    print("OK: fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm all import cleanly.")

    if joyomni_ops.has_fp8():
        print("OK: FP8 kernels available.")
    else:
        print("NOTE: FP8 kernels not available (built with JOYOMNI_OPS_NO_FP8=1, or cutlass missing).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
