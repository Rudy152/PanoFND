@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_qwen_pilot.ps1"
echo.
echo The pilot run has stopped. Keep this window open and tell Codex it is finished.
pause
endlocal
