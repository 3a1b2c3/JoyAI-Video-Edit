#!/bin/bash
# Rebuild joyomni_ops with FP8 support enabled
#
# FP8 requires:
#   - CUDA >= 12.8 (for Blackwell support)
#   - NVIDIA cutlass library
#   - PyTorch with CUDA support
#
# If build fails or fp8 is not needed, use:
#   JOYOMNI_OPS_NO_FP8=1 python -m pip install -e . --no-build-isolation

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

LOG_FILE="$HERE/build.log"
: > "$LOG_FILE"  # fresh log each run; pre-build diagnostics + pip install output both land here
log() { echo "$@" | tee -a "$LOG_FILE"; }

# Build against whichever interpreter will actually run inference. A .so built
# for one CPython version/ABI is silently unimportable under another: pip
# install then fails to register a real package, and `import joyomni_ops`
# falls back to resolving a bare namespace package from the source checkout
# instead (ImportError: "... (unknown location)").
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: interpreter '$PYTHON' not found."
    echo "  Set PYTHON=/path/to/python3.x to point at the venv you run inference with."
    exit 1
fi

echo "=========================================="
echo "Building joyomni_ops with FP8 support"
echo "=========================================="
echo ""
echo "--- pre-build diagnostics (also written to $LOG_FILE) ---"
{
    echo "-- interpreter / ABI --"
    echo "executable:  $("$PYTHON" -c 'import sys; print(sys.executable)')"
    echo "version:     $("$PYTHON" --version 2>&1)"
    echo "EXT_SUFFIX (target ABI tag): $("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    echo "SOABI:                       $("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
    echo "platform tag:                $("$PYTHON" -c 'import sysconfig; print(sysconfig.get_platform())')"
    echo ""
    echo "pip:         $("$PYTHON" -m pip --version 2>&1)"
    echo "setuptools:  $("$PYTHON" -c 'import setuptools; print(setuptools.__version__)' 2>/dev/null || echo '<not installed>')"
    echo ""
    echo "-- python packages --"
    "$PYTHON" -c "
for name in ('torch', 'torchvision', 'torchaudio', 'ninja', 'wheel', 'packaging'):
    try:
        mod = __import__(name)
        print(f'{name}: {getattr(mod, \"__version__\", \"<no __version__>\")}')
    except ImportError:
        print(f'{name}: NOT INSTALLED')
"
    torch_cuda=$("$PYTHON" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "")
    echo "torch.version.cuda: ${torch_cuda:-<torch import failed>}"
    echo ""
    echo "-- build tools --"
    echo "gcc:    $(gcc --version 2>/dev/null | head -n 1 || echo '<not found>')"
    echo "g++:    $(g++ --version 2>/dev/null | head -n 1 || echo '<not found>')"
    echo "cmake:  $(cmake --version 2>/dev/null | head -n 1 || echo '<not found>')"
    echo "ninja:  $(ninja --version 2>/dev/null || echo '<not found>')"
    echo "make:   $(make --version 2>/dev/null | head -n 1 || echo '<not found>')"
    echo ""
    echo "-- GPU / driver --"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null || echo "MISSING: nvidia-smi not found"
    echo ""
    echo "-- OS --"
    uname -srm 2>/dev/null
    [ -f /etc/os-release ] && grep -E '^(NAME|VERSION)=' /etc/os-release
    echo ""
    echo "-- git / submodules --"
    echo "HEAD: $(git rev-parse --short HEAD 2>/dev/null) ($(git branch --show-current 2>/dev/null))"
    git submodule status 2>/dev/null
} 2>&1 | tee -a "$LOG_FILE" | sed 's/^/  /'
echo ""

# Check CUDA is available
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found."
    echo ""
    echo "Ensure CUDA is installed and in PATH:"
    echo "  source /usr/local/cuda-12.x/setup_env.sh  (or your CUDA path)"
    echo "  export PATH=/usr/local/cuda-12.x/bin:\$PATH"
    echo ""
    echo "Alternatively, build without FP8:"
    echo "  JOYOMNI_OPS_NO_FP8=1 python -m pip install -e . --no-build-isolation"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' || echo "unknown")
echo "[1/4] CUDA version: $CUDA_VERSION"
if [[ "$CUDA_VERSION" < "12.8" ]]; then
    echo "⚠️  CUDA < 12.8 — FP8 will work but without native Blackwell (sm_120) SASS."
    echo "    (Uses sm_90 PTX JIT instead; consider upgrading for full performance.)"
fi
echo ""

# Clone cutlass if not present
echo "[2/4] Checking NVIDIA cutlass..."
if [ ! -d "cutlass" ]; then
    echo "  Cloning cutlass (required for fp8_gemm.cu)..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
    if [ ! -d "cutlass/include" ]; then
        echo "ERROR: cutlass clone failed"
        exit 1
    fi
    echo "  ✓ Cloned"
else
    if [ ! -f "cutlass/include/cutlass/cutlass.h" ]; then
        echo "ERROR: cutlass present but broken (missing cutlass.h)"
        echo "  Try: rm -rf cutlass && bash build.sh"
        exit 1
    fi
    echo "  ✓ Already present"
fi
echo ""

# Clean old build
echo "[3/4] Cleaning old build artifacts..."
rm -rf build dist *.egg-info
echo "  ✓ Cleaned"
echo ""

# Build with FP8
echo "[4/4] Building joyomni_ops (fp8 enabled)..."
export JOYOMNI_OPS_CUTLASS_DIR="$HERE/cutlass"
unset JOYOMNI_OPS_NO_FP8 2>/dev/null || true

# setup.py reads JOYOMNI_OPS_NO_FP8 straight from the environment (see setup.py) --
# if it's still truthy here (e.g. left over from an earlier `export
# JOYOMNI_OPS_NO_FP8=1` workaround in this same shell session), fp8_gemm.cu silently
# gets excluded from the build and joyomni_ops._C ends up missing fp8_scaled_mm /
# sgl_per_token_quant_fp8 with no build error -- the exact bug that cost real time to
# diagnose. Fail loudly instead of building a silently-crippled extension.
log "JOYOMNI_OPS_NO_FP8 (post-unset, must be empty): '${JOYOMNI_OPS_NO_FP8:-}'"
log "JOYOMNI_OPS_CUTLASS_DIR: $JOYOMNI_OPS_CUTLASS_DIR"
if [ -n "${JOYOMNI_OPS_NO_FP8:-}" ]; then
    echo "ERROR: JOYOMNI_OPS_NO_FP8 is still set to '${JOYOMNI_OPS_NO_FP8}' after unset -- something"
    echo "  (a shell rc file?) is re-exporting it. This build would silently skip FP8."
    exit 1
fi

# Cutlass FP8 template compiles can legitimately run 10+ min per file with zero output
# in between, so there's no safe timeout to kill on. Log compiler-process CPU%/elapsed
# time (rising %CPU + growing ETIME on cicc/ptxas = genuinely working, not hung) plus a
# coarse "N of M kernel .o files done" counter every 30s, so progress is visible live
# via `tail -f build.log` without needing to interrupt an in-progress build to check.
KERNEL_COUNT=$(ls csrc/*.cu 2>/dev/null | wc -l)
(
    while true; do
        sleep 30
        {
            echo "--- watchdog $(date +%H:%M:%S) ---"
            ps -eo pid,pcpu,stat,etime,cmd 2>/dev/null | grep -E "cicc|ptxas|cc1plus|nvcc" | grep -v grep
            DONE=$(ls /tmp/tmp*.build-temp/csrc/*.o 2>/dev/null | wc -l)
            echo "kernel objects compiled: $DONE / $KERNEL_COUNT"
            ls -la /tmp/tmp*.build-temp/csrc/*.o 2>/dev/null
        } >> "$LOG_FILE"
    done
) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null' EXIT

if ! VERBOSE=1 "$PYTHON" -m pip install -e . --no-build-isolation --force-reinstall -v 2>&1 | tee -a "$LOG_FILE"; then
    echo ""
    echo "ERROR: Build failed. See build.log for details."
    echo ""
    echo "Common fixes:"
    echo "  1. Check CUDA is in PATH:  nvcc --version"
    echo "  2. Ensure cutlass is valid:  test -f cutlass/include/cutlass/cutlass.h"
    echo "  3. Check pip output (last 50 lines):"
    echo ""
    tail -50 "$LOG_FILE" | sed 's/^/    /'
    echo ""
    echo "If FP8 is not needed, build without it:"
    echo "  JOYOMNI_OPS_NO_FP8=1 $PYTHON -m pip install -e . --no-build-isolation"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "=========================================="
echo ""

# Verify the install actually resolves to the real package under this
# interpreter, not a namespace package pieced together from source dirs on
# sys.path (the failure mode a version/ABI mismatch produces).
echo "Verifying real package install..."
if ! "$PYTHON" -c "
import joyomni_ops
if not getattr(joyomni_ops, '__file__', None):
    raise ImportError('joyomni_ops resolved as a namespace package (no __file__) -- not a real install')
from joyomni_ops import fused_norm_scale_shift, fused_qk_norm_rope_3d_paired, rmsnorm
print(f'  ✓ joyomni_ops loaded from {joyomni_ops.__file__}')
"; then
    echo ""
    echo "ERROR: joyomni_ops installed but not importable as a real package under $PYTHON."
    echo "  Likely a stray joyomni_ops/ directory earlier on sys.path shadowing the install,"
    echo "  or the .so's Python ABI tag doesn't match this interpreter."
    exit 1
fi
echo ""

# Verify FP8 functions are available. NOTE: pybind.cpp uses TORCH_LIBRARY (dispatcher
# registration via torch.ops.load_library at dlopen time), not PYBIND11_MODULE -- there
# is no joyomni_ops._C Python submodule to import from, and _C.so legitimately has no
# PyInit__C symbol by design. The real functions are the top-level wrappers __init__.py
# defines around torch.ops.joyomni_ops.*; has_fp8() is what actually confirms the FP8
# op got registered (the wrapper functions exist unconditionally either way).
echo "Verifying FP8 functions..."
if "$PYTHON" -c "
from joyomni_ops import fp8_scaled_mm, sgl_per_token_quant_fp8, has_fp8
assert has_fp8(), 'fp8_scaled_mm exists as a Python wrapper but is not registered with torch.ops -- FP8 kernels were not compiled in'
print('  ✓ fp8_scaled_mm available')
print('  ✓ sgl_per_token_quant_fp8 available')
" 2>&1; then
    echo ""
    echo "✅ FP8 support verified!"
    exit 0
else
    echo ""
    echo "⚠️  FP8 functions not available after build."
    echo ""
    echo "This may happen if:"
    echo "  - cutlass headers are missing"
    echo "  - CUDA compiler failed silently"
    echo ""
    log "=== Auto-diagnostics (also appended to $LOG_FILE) ==="
    echo ""
    echo "-- tail -100 build.log --"
    tail -100 "$LOG_FILE" | sed 's/^/  /'
    {
        echo ""
        echo "-- cutlass headers --"
        if [ -d cutlass/include/cutlass ] && [ "$(ls -A cutlass/include/cutlass 2>/dev/null)" ]; then
            echo "OK: cutlass/include/cutlass/ present ($(ls cutlass/include/cutlass | wc -l) entries)"
        else
            echo "MISSING/EMPTY: cutlass/include/cutlass/ -- submodule likely not checked out."
            echo "Fix: git submodule update --init --recursive"
        fi
        echo ""
        echo "-- CUDA toolchain --"
        which nvcc || echo "MISSING: nvcc not on PATH"
        nvcc --version 2>/dev/null | tail -n 4
        echo "torch.version.cuda: $("$PYTHON" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '<torch import failed>')"
        echo ""
        # Specifically for "ImportError: dynamic module does not define module export
        # function (PyInit__C)": the .so built/linked/copied fine, but the symbol
        # Python's import machinery looks for isn't in it. Narrows down whether the
        # module-name macro is wrong, the symbol got stripped/hidden, or a stale .so
        # elsewhere on disk is shadowing the fresh build.
        echo "-- PyInit__C diagnostics (for 'dynamic module does not define module export function') --"
        echo "pybind module macro:"
        PYBIND_MACRO="$(grep -n "PYBIND11_MODULE\|TORCH_EXTENSION_NAME" csrc/pybind.cpp 2>&1)"
        if [ -z "$PYBIND_MACRO" ]; then
            echo "  MISSING: csrc/pybind.cpp has no PYBIND11_MODULE/TORCH_EXTENSION_NAME text at all --"
            echo "  this is why no PyInit__C symbol is ever generated, however cleanly the rest compiles."
            echo "  Full file (so the actual registration mechanism, if any, is visible):"
            sed 's/^/    /' csrc/pybind.cpp
        else
            echo "$PYBIND_MACRO" | sed 's/^/  /'
        fi
        SO_PATH="$HERE/joyomni_ops/_C$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' 2>/dev/null)"
        echo "expected .so: $SO_PATH"
        ls -la "$SO_PATH" 2>&1 | sed 's/^/  /'
        echo "symbols (nm -D):"
        nm -D "$SO_PATH" 2>&1 | grep -i "PyInit\|undefined" | sed 's/^/  /'
        echo "symbols (objdump -T, catches what nm -D sometimes misses):"
        objdump -T "$SO_PATH" 2>&1 | grep -i PyInit | sed 's/^/  /'
        # If PyInit__C is present here (full symtab) but absent from -D/-T above, the
        # function exists but was compiled/linked as hidden-visibility -- present in
        # the binary, just never exported for dynamic linking (Python's importer only
        # looks at the dynamic table). If it's absent here too, pybind.cpp genuinely
        # never generated it (macro missing/broken), a different root cause.
        echo "symbols (nm without -D, full symtab -- present-but-hidden vs. never-generated):"
        nm "$SO_PATH" 2>&1 | grep -i PyInit | sed 's/^/  /'
        echo "pybind.cpp macro / visibility mentions:"
        grep -n "PYBIND11_MODULE\|TORCH_EXTENSION_NAME\|visibility" csrc/pybind.cpp 2>&1 | sed 's/^/  /'
        echo "setup.py -fvisibility flags:"
        grep -n "fvisibility" setup.py 2>&1 | sed 's/^/  /'
        echo "other _C*.so files on disk (possible stale/shadowing copy):"
        # Scoped to $HERE + this interpreter's actual site-packages dirs, NOT a
        # filesystem-wide `find /` -- that can hang for minutes on a shared box and
        # has repeatedly stalled this diagnostic block out from under people.
        SEARCH_DIRS="$HERE $("$PYTHON" -c "import site; print(' '.join(site.getsitepackages() + [site.getusersitepackages()]))" 2>/dev/null || true)"
        find $SEARCH_DIRS -maxdepth 6 -name "_C*.so" -path "*joyomni_ops*" 2>/dev/null | sed 's/^/  /'
    } 2>&1 | tee -a "$LOG_FILE" | sed 's/^/  /'
    echo ""
    echo "Workaround (build without FP8, only if the above doesn't point to a fixable cause):"
    echo "  JOYOMNI_OPS_NO_FP8=1 $PYTHON -m pip install -e . --no-build-isolation --force-reinstall"
    exit 1
fi
