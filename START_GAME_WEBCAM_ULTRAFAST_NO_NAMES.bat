@echo off
cd /d "%~dp0"
echo ULTRAFAST NO-NAMES MODE: Face ID and evidence are disabled.
py main.py play --webcam --no-flight --no-face-id --no-evidence
pause
