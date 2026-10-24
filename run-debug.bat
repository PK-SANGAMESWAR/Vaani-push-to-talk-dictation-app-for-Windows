@echo off
rem Runs Vaani with a console window and verbose logs (for debugging).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m vaani -v
pause
