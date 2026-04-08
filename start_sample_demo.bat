@echo off
setlocal
cd /d "%~dp0"

py -u sample_demo_app.py
if errorlevel 1 (
  echo.
  echo Falling back to python...
  python -u sample_demo_app.py
)

pause
