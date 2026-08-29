@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "CKPT_DIR=%SCRIPT_DIR%deploy\deps\checkpoints"
if not exist "!CKPT_DIR!" mkdir "!CKPT_DIR!"

set "YOLO_PATH=!CKPT_DIR!\yolov8n.onnx"

if exist "!YOLO_PATH!" (
  echo YOLOv8n already exists at !YOLO_PATH!
  exit /b 0
)

echo Exporting YOLOv8n to ONNX...

set "TEMP_PY=%TEMP%\yolo_export_temp.py"

(
  echo import os, sys
  echo script_dir = r"%SCRIPT_DIR:\=/%"
  echo ckpt_dir = os.path.join(script_dir, "deploy", "deps", "checkpoints"^)
  echo os.makedirs(ckpt_dir, exist_ok=True^)
  echo yolo_path = os.path.join(ckpt_dir, "yolov8n.onnx"^)
  echo.
  echo try:
  echo     from ultralytics import YOLO
  echo     print("Loading YOLOv8n..."^)
  echo     model = YOLO("yolov8n.pt"^)
  echo     print("Exporting to ONNX..."^)
  echo     model.export(format="onnx", imgsz=640^)
  echo     print(f"OK: {yolo_path}"^)
  echo except ImportError:
  echo     print("Installing ultralytics..."^)
  echo     os.system("pip install ultralytics"^)
  echo     from ultralytics import YOLO
  echo     model = YOLO("yolov8n.pt"^)
  echo     model.export(format="onnx", imgsz=640^)
  echo     print(f"OK: {yolo_path}"^)
  echo except Exception as e:
  echo     print(f"ERROR: {e}"^)
  echo     sys.exit(1^)
) > "!TEMP_PY!"

python "!TEMP_PY!"
set "RETCODE=%ERRORLEVEL%"
del /q "!TEMP_PY!" 2>nul
exit /b !RETCODE!
