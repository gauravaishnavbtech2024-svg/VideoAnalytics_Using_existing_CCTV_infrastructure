@echo off
title FaceTrace - ATM Face Finder
echo ===================================================
echo Starting FaceTrace ATM Face Finder (Kivy Desktop Console)...
echo ===================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python facetrace_app.py
pause
