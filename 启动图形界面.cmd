@echo off
setlocal
cd /d "%~dp0"
python -c "import tkinter, numpy, requests, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Python or a required package could not be loaded.
    echo Install Python 3.10+ with Tk and run: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" pythonw.exe gui.py
