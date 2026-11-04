@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py launch_app.py
) else (
    python launch_app.py
)
