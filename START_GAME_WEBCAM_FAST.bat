@echo off
cd /d "%~dp0"
echo FAST WITH NAMES: Face ID runs before the game, then the active game stream stays lightweight.
py main.py play --webcam --no-flight
pause
