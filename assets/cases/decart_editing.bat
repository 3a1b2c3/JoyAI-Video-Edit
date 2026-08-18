@echo off
setlocal enableextensions
REM ==========================================================================
REM [lucy-restyle-2, QUEUE/BATCH API] Decart on editing.mp4, using assets\image.png
REM as the reference image. Switched from decart_long.py (realtime/WebRTC) because
REM the realtime path doesn't return a frame-accurate output (194 frames back vs
REM 335 in the source -- render-rate/network jitter, not a skip-count issue).
REM decart_batch.py uploads the whole clip over HTTPS and processes it server-side,
REM so output length should track the source instead of live-stream best-effort.
REM Note: batch API takes reference_image OR prompt, never both -- REF wins here,
REM so the earlier motion-language prompt doesn't apply to this call.
REM Needs: DECART_API_KEY set.
REM Output: scope\out\decart_cases_editing.mp4
REM ==========================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "DECART_MODEL=lucy-restyle-2"
set "DECART_TAG=editing"
set "DECART_CONTROL=%HERE%\editing.mp4"
set "DECART_REF=%HERE%\..\image.png"
"C:\workspace\world\decart_venv\Scripts\python.exe" "C:\workspace\world\decart_batch.py" "%HERE%"
