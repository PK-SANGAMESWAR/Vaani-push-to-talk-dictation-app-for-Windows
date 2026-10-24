@echo off
rem Starts Vaani in the background (tray icon, no console window).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m vaani
