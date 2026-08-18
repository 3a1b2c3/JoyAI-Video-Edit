@echo off
setlocal enableextensions
REM ==========================================================================
REM [lucy-restyle-2, QUEUE/BATCH API] Decart on wind.mp4, using assets\image.png
REM as the reference image. Same pattern as decart_mattress.bat/decart_editing.bat
REM (frame-accurate length via the batch API).
REM Note: batch API takes reference_image OR prompt, never both -- REF wins here.
REM Needs: DECART_API_KEY set.
REM Output: scope\out\decart_cases_wind.mp4
REM ==========================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "DECART_MODEL=lucy-restyle-2"
set "DECART_TAG=wind"
set "DECART_CONTROL=%HERE%\wind.mp4"
set "DECART_REF=%HERE%\..\image.png"
"C:\workspace\world\decart_venv\Scripts\python.exe" "C:\workspace\world\decart_batch.py" "%HERE%"
