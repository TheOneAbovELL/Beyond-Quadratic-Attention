@echo off
rem Double-click to start the Beyond Quadratic Attention lab (see start.py for what it does).
title Beyond Quadratic Attention
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 start.py %*
) else (
    python start.py %*
)
if errorlevel 1 pause
