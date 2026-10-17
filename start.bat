@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 run.py %*
) else (
    python run.py %*
)
if errorlevel 1 (
    echo.
    echo Install Python 3.11 or newer if Python was not found.
    pause
)
