# Building joyomni_ops on Windows

## Prerequisites

- **Python 3.10** (with venv activated)
- **CUDA Toolkit 12.8** - https://developer.nvidia.com/cuda-toolkit
- **Visual Studio 2022 Community** with C++ tools
- **Git** (for cloning, if needed)

## Quick Build

### Step 1: Open Developer Command Prompt

Press `Win + R` and run:
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
```

This sets up MSVC environment variables automatically.

### Step 2: Navigate and Install Dependencies

```cmd
cd C:\workspace\world\JoyAI-Video-Edit\deploy\joyomni_ops
..\..\..\.venv\Scripts\python.exe -m pip install setuptools wheel
```

### Step 3: Build

```cmd
..\..\..\.venv\Scripts\python.exe setup.py build_ext --inplace
```

### Step 4: Verify

```cmd
..\..\..\.venv\Scripts\python.exe -c "import torch; torch.ops.load_library('./joyomni_ops/_C.cpython-310-x86_64.pyd'); print('✅ OK')"
```

## Using the Build Script

Instead of manual steps, use the automated script:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
C:\workspace\world\JoyAI-Video-Edit\rebuild_joyomni.bat
```

The script:
1. Fixes setup.py CUDA path to 12.8
2. Cleans previous build
3. Compiles CUDA kernels with nvcc
4. Links C++ code with cl.exe (MSVC)
5. Verifies the .pyd was created

## Important Notes

### Must Use Developer Command Prompt

The Visual Studio Developer Command Prompt sets up:
- `cl.exe` (MSVC C++ compiler) in PATH
- Windows SDK include/lib paths
- Environment variables for building

Without it, you'll get:
```
Cannot find compiler 'cl.exe' in PATH
Cannot open include file: 'corecrt.h'
```

### setuptools Required

If you see `ModuleNotFoundError: No module named 'setuptools'`:

```cmd
..\..\..\.venv\Scripts\python.exe -m pip install setuptools
```

### CUDA Path

The script automatically fixes `setup.py` to use CUDA 12.8:
```cmd
powershell -Command "(Get-Content setup.py) -replace 'v12\.4', 'v12.8' | Set-Content setup.py"
```

If you have a different CUDA version, edit manually.

## Output

Successful build produces:
```
joyomni_ops\_C.cpython-310-x86_64.pyd  (~10 MB)
```

The .pyd (Python Dynamic Library) is a compiled CUDA extension that provides:
- `torch.ops.joyomni_ops.fused_norm_scale_shift`
- `torch.ops.joyomni_ops.fused_qk_norm_rope_3d_paired`
- `torch.ops.joyomni_ops.rmsnorm`

## Testing

Once built, test locally:

```powershell
cd C:\workspace\world\JoyAI-Video-Edit
bash run_inference.sh
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `cl.exe not found` | Use Developer Command Prompt |
| `Cannot open include file: corecrt.h` | Same as above |
| `No module named setuptools` | `python -m pip install setuptools` |
| `CUDA 12.4 not found` | Script auto-fixes to 12.8; verify path |
| `PyInit__C not found` | Normal - uses torch.ops instead |

## Linux/Horde Build

For building on horde (A40, 48GB), see [BUILD_JOYOMNI_OPS.md](BUILD_JOYOMNI_OPS.md).

Key difference: Linux uses .so (shared object), Windows uses .pyd (Python Dynamic Library).
