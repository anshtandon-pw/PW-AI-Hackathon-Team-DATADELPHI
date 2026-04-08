@echo off
setlocal
cd /d "%~dp0"

echo Running sample training pipeline...
py -u run_sample_training_pipeline.py
if errorlevel 1 (
  echo.
  echo Falling back to python for the pipeline...
  python -u run_sample_training_pipeline.py
)

if errorlevel 1 (
  echo.
  echo The sample pipeline failed. Press any key to close this window.
  pause >nul
  exit /b 1
)

echo.
echo Starting sample demo...
py -u sample_demo_app.py
if errorlevel 1 (
  echo.
  echo Falling back to python for the demo...
  python -u sample_demo_app.py
)

pause
