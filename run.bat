@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Launching PointCloud Downsampler...
py -3.11 main.py

if errorlevel 1 (
    echo.
    echo [ERROR] App failed to start.
    echo If dependencies are missing, run:
    echo   py -3.11 -m pip install -r requirements.txt
)

pause
