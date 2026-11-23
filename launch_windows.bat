@echo off
cd /d "%~dp0"
py main.py
if errorlevel 1 (
  echo.
  echo If dependencies are missing, run: py -m pip install -r requirements.txt
  pause
)
