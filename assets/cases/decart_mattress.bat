@echo off
setlocal enableextensions
REM ==========================================================================
REM [lucy-restyle-2, QUEUE/BATCH API] Decart on mattress.mp4, using assets\image.png
REM as the reference image. Switched from decart_long.py (realtime/WebRTC) --
REM see decart_editing.bat for why (frame-accurate length via the batch API).
REM Note: batch API takes reference_image OR prompt, never both -- REF wins here.
REM Needs: DECART_API_KEY set.
REM Output: scope\out\decart_cases_mattress.mp4
REM ==========================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "DECART_MODEL=lucy-restyle-2"
set "DECART_TAG=mattress"
set "DECART_CONTROL=%HERE%\mattress.mp4"
set "DECART_REF=%HERE%\..\image.png"
"C:\workspace\world\decart_venv\Scripts\python.exe" "C:\workspace\world\decart_batch.py" "%HERE%"
